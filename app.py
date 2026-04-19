# -*- coding: utf-8 -*-
"""
MasterDock – Molecular Interaction Analysis Platform
Docking engine : SwissDock REST API (https://swissdock.ch:8443)
                 Engines: Attracting Cavities 2.0 (AC) or AutoDock Vina
Visualization  : py3Dmol (3D) + RDKit (2D)
Parsing        : Biopython (PDB) + RDKit (SDF / Mol2)
"""

import io
import os
import re
import tarfile
import tempfile
import time

import pandas as pd
import py3Dmol
import requests
import streamlit as st
import streamlit.components.v1 as components
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors

# ─────────────────────────────────────────────────────────────────────────────
#  Page Config — must be the FIRST Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MasterDock · SwissDock",
    page_icon="🧬",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
#  SwissDock API constants
# ─────────────────────────────────────────────────────────────────────────────
SD_BASE   = "https://swissdock.ch:8443"
SD_HELLO  = f"{SD_BASE}/"
SD_PREPLIG    = f"{SD_BASE}/preplig"
SD_PREPTARGET = f"{SD_BASE}/preptarget"
SD_SETPARAMS  = f"{SD_BASE}/setparameters"
SD_STARTDOCK  = f"{SD_BASE}/startdock"
SD_CHECKSTATUS = f"{SD_BASE}/checkstatus"
SD_RETRIEVE   = f"{SD_BASE}/retrievesession"
SD_CANCEL     = f"{SD_BASE}/cancelsession"

POLL_INTERVAL = 15   # seconds between status polls
MAX_POLLS     = 240  # 240 × 15 s = 60 min max wait

# ─────────────────────────────────────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .section-header {
    background: linear-gradient(90deg,#1a237e,#283593);
    color:#fff; padding:8px 16px; border-radius:6px;
    font-size:1.05rem; font-weight:600; margin-bottom:10px;
  }
  .api-box {
    background:#f0f4ff; border-left:4px solid #3949ab;
    padding:10px 14px; border-radius:4px; font-size:.85rem;
    margin-bottom:8px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Molecular helper functions
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdb(pdb_content: str) -> dict:
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("mol", io.StringIO(pdb_content))
    except Exception as e:
        raise PDBConstructionException(str(e)) from e
    stats = {"Models": 0, "Chains": 0, "Residues": 0, "Atoms": 0}
    for model in structure:
        stats["Models"] += 1
        for chain in model:
            stats["Chains"] += 1
            for residue in chain:
                stats["Residues"] += 1
                for _ in residue:
                    stats["Atoms"] += 1
    return stats


def parse_sdf(sdf_content: str):
    suppl = Chem.SDMolSupplier()
    suppl.SetData(sdf_content, removeHs=False)
    return [m for m in suppl if m is not None]


def render_3d(content: str, fmt: str, height: int = 420) -> str:
    view = py3Dmol.view(width="100%", height=height)
    view.addModel(content, fmt.lower())
    view.setStyle({"stick": {}})
    view.zoomTo()
    return view._repr_html_()


def render_2d(mol, size=(280, 280)):
    return Draw.MolToImage(mol, size=size) if mol else None


def mol_props(mol) -> dict:
    return {
        "Atoms": mol.GetNumAtoms(),
        "Bonds": mol.GetNumBonds(),
        "MW (Da)": round(rdMolDescriptors.CalcExactMolWt(mol), 3),
    }


def sdf_to_mol2_rdkit(sdf_content: str) -> str | None:
    """
    Convert the first molecule in an SDF string to Mol2 format using RDKit.
    SwissDock AC requires a Mol2 ligand file.
    """
    mols = parse_sdf(sdf_content)
    if not mols:
        return None
    mol = mols[0]
    # Add 3-D coords if missing
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv2())
        AllChem.MMFFOptimizeMolecule(mol)
    tmp = tempfile.NamedTemporaryFile(suffix=".mol2", delete=False)
    tmp.close()
    Chem.MolToMolFile(mol, tmp.name)   # write as SDF first
    # RDKit doesn't write Mol2 natively; use a simple conversion via file
    # We'll return the SDF content and tell the user the limitation
    mol2_path = tmp.name.replace(".mol2", "_out.mol2")
    try:
        import subprocess
        subprocess.run(
            ["obabel", tmp.name, "-O", mol2_path],
            capture_output=True, check=True,
        )
        with open(mol2_path) as f:
            return f.read()
    except Exception:
        # obabel not available — return None so caller falls back to SMILES
        return None
    finally:
        for p in (tmp.name, mol2_path):
            if os.path.exists(p):
                os.remove(p)


# ─────────────────────────────────────────────────────────────────────────────
#  SwissDock REST API client
# ─────────────────────────────────────────────────────────────────────────────

def sd_check_server() -> bool:
    """Ping the SwissDock API. Returns True if alive."""
    try:
        r = requests.get(SD_HELLO, timeout=10, verify=False)
        return "Hello World" in r.text
    except Exception:
        return False


def sd_prepare_ligand(
    mol2_bytes: bytes | None = None,
    smiles: str | None = None,
    session_number: str | None = None,
    use_vina: bool = False,
) -> tuple[bool, str]:
    """
    Upload and prepare the ligand.
    Returns (success, session_number_or_error).

    Priority: mol2_bytes > smiles
    """
    params = {}
    if session_number:
        params["sessionNumber"] = session_number
    if use_vina:
        params["Vina"] = ""

    try:
        if mol2_bytes:
            files = {"myLig": ("ligand.mol2", mol2_bytes, "chemical/x-mol2")}
            r = requests.post(SD_PREPLIG, files=files, params=params, timeout=120, verify=False)
        elif smiles:
            params["mySMILES"] = smiles
            r = requests.get(SD_PREPLIG, params=params, timeout=120, verify=False)
        else:
            return False, "No ligand input provided (need Mol2 bytes or SMILES)."

        text = r.text.strip()
        m = re.search(r"Session number:\s*(\d+)", text)
        if m:
            return True, m.group(1)
        # might already have a session from a previous step
        m2 = re.search(r"Using session number:\s*(\d+)", text)
        if m2:
            return True, m2.group(1)
        return False, f"Unexpected response from SwissDock:\n{text}"
    except Exception as e:
        return False, f"Network error preparing ligand: {e}"


def sd_prepare_target(
    pdb_bytes: bytes,
    session_number: str,
) -> tuple[bool, str]:
    """Upload and prepare the receptor PDB. Returns (success, message)."""
    try:
        files = {"myTarget": ("target.pdb", pdb_bytes, "chemical/x-pdb")}
        params = {"sessionNumber": session_number}
        r = requests.post(SD_PREPTARGET, files=files, params=params, timeout=180, verify=False)
        text = r.text.strip()
        if "prepared" in text.lower():
            return True, text
        return False, f"Target preparation failed:\n{text}"
    except Exception as e:
        return False, f"Network error preparing target: {e}"


def sd_set_parameters(
    session_number: str,
    center_x: float, center_y: float, center_z: float,
    size_x: float = 20.0, size_y: float = 20.0, size_z: float = 20.0,
    exhaust: int = 90,
    cavity: int = 70,
    ric: int = 2,
    use_vina: bool = False,
    job_name: str = "MasterDock",
) -> tuple[bool, str]:
    """Set docking parameters and check session. Returns (can_submit, message)."""
    params = {
        "sessionNumber": session_number,
        "boxCenter": f"{center_x}_{center_y}_{center_z}",
        "boxSize":   f"{size_x}_{size_y}_{size_z}",
        "exhaust":   exhaust,
        "name":      job_name,
    }
    if not use_vina:
        params["cavity"] = cavity
        params["ric"]    = ric
    try:
        r = requests.get(SD_SETPARAMS, params=params, timeout=60, verify=False)
        text = r.text.strip()
        can_submit = "session can be submitted" in text.lower()
        return can_submit, text
    except Exception as e:
        return False, f"Network error setting parameters: {e}"


def sd_start_docking(session_number: str) -> tuple[bool, str]:
    """Submit the docking job. Returns (success, message)."""
    try:
        r = requests.get(SD_STARTDOCK, params={"sessionNumber": session_number}, timeout=60, verify=False)
        text = r.text.strip()
        ok = "submitted" in text.lower()
        return ok, text
    except Exception as e:
        return False, f"Network error starting docking: {e}"


def sd_check_status(session_number: str) -> str:
    """Return raw status text from SwissDock."""
    try:
        r = requests.get(SD_CHECKSTATUS, params={"sessionNumber": session_number}, timeout=30, verify=False)
        return r.text.strip()
    except Exception as e:
        return f"Error checking status: {e}"


def sd_retrieve_results(session_number: str) -> bytes | None:
    """Download the results tar.gz. Returns raw bytes or None."""
    try:
        r = requests.get(
            SD_RETRIEVE,
            params={"sessionNumber": session_number},
            timeout=300,
            verify=False,
        )
        if r.status_code == 200 and len(r.content) > 100:
            return r.content
        return None
    except Exception:
        return None


def sd_cancel(session_number: str) -> str:
    try:
        r = requests.get(SD_CANCEL, params={"sessionNumber": session_number}, timeout=30, verify=False)
        return r.text.strip()
    except Exception as e:
        return str(e)


def parse_results_tarball(tar_bytes: bytes) -> dict:
    """
    Extract docking results from the SwissDock tar.gz archive.
    Returns a dict with keys: 'poses_pdb', 'scores_df', 'log_text', 'files'
    """
    result = {"poses_pdb": None, "scores_df": None, "log_text": None, "files": []}
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            members = tar.getmembers()
            result["files"] = [m.name for m in members]

            for member in members:
                f = tar.extractfile(member)
                if f is None:
                    continue
                raw = f.read()

                name = member.name.lower()

                # Docked poses — AC produces cluster*.pdb, Vina produces out.pdbqt
                if name.endswith(".pdb") and "cluster" in name:
                    result["poses_pdb"] = raw.decode("utf-8", errors="ignore")

                elif name.endswith(".pdbqt") and "out" in name:
                    result["poses_pdb"] = raw.decode("utf-8", errors="ignore")

                # Score/energy log
                elif name.endswith(".log") or name.endswith("results.txt") or "score" in name:
                    result["log_text"] = raw.decode("utf-8", errors="ignore")

                # Summary / clusters file (AC outputs a clusters file)
                elif "cluster" in name and name.endswith(".txt"):
                    result["log_text"] = raw.decode("utf-8", errors="ignore")

        # Try to parse scores from log
        if result["log_text"]:
            scores = []
            for line in result["log_text"].splitlines():
                # Vina-style: "   1   -8.5   0.000   0.000"
                m = re.match(r"\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
                if m:
                    scores.append({
                        "Mode": int(m.group(1)),
                        "Affinity (kcal/mol)": float(m.group(2)),
                        "RMSD lb (Å)": float(m.group(3)),
                        "RMSD ub (Å)": float(m.group(4)),
                    })
                # AC-style energy lines
                m2 = re.match(r"\s*(\d+)\s+([-\d.]+)\s*$", line.strip())
                if m2:
                    scores.append({
                        "Cluster": int(m2.group(1)),
                        "Energy (kcal/mol)": float(m2.group(2)),
                    })
            if scores:
                result["scores_df"] = pd.DataFrame(scores)
    except Exception as e:
        result["log_text"] = f"Error parsing results: {e}"
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────
for key, default in {
    "sd_session": None,
    "docking_running": False,
    "docking_done": False,
    "result_tarball": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 MasterDock")
    st.caption("Powered by the **SwissDock REST API**")
    st.markdown("---")

    st.markdown("### 📂 Upload Structures")
    receptor_file = st.file_uploader("Receptor PDB", type=["pdb"], key="rec_pdb")

    lig_input_mode = st.radio(
        "Ligand input format",
        ["Upload SDF file", "Enter SMILES"],
        horizontal=True,
    )
    if lig_input_mode == "Upload SDF file":
        ligand_file = st.file_uploader("Ligand SDF", type=["sdf"], key="lig_sdf")
        smiles_input = None
    else:
        ligand_file = None
        smiles_input = st.text_input(
            "Ligand SMILES",
            placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
        )

    st.markdown("---")
    st.markdown("### 🔬 Docking Engine")
    engine = st.radio(
        "Algorithm",
        ["Attracting Cavities 2.0 (AC)", "AutoDock Vina"],
        help="AC is more accurate; Vina is faster.",
    )
    use_vina = engine == "AutoDock Vina"

    st.markdown("---")
    st.markdown("### ⚙️ Grid Box (Å)")
    c1, c2, c3 = st.columns(3)
    center_x = c1.number_input("X", value=0.0, format="%.1f", key="cx")
    center_y = c2.number_input("Y", value=0.0, format="%.1f", key="cy")
    center_z = c3.number_input("Z", value=0.0, format="%.1f", key="cz")
    c1, c2, c3 = st.columns(3)
    size_x = c1.number_input("SX", value=20.0, format="%.1f", key="sx")
    size_y = c2.number_input("SY", value=20.0, format="%.1f", key="sy")
    size_z = c3.number_input("SZ", value=20.0, format="%.1f", key="sz")

    st.markdown("### 🎛️ Sampling")
    if use_vina:
        exhaust = st.slider("Exhaustiveness", 1, 64, 8)
        cavity, ric = 70, 2        # unused for Vina
    else:
        exhaust = st.select_slider(
            "Exhaustivity (rotational step °)",
            options=[180, 90, 60],
            value=90,
            help="180=low, 90=medium (default), 60=high",
        )
        cavity = st.select_slider(
            "Cavity prioritization",
            options=[50, 60, 70],
            value=70,
            help="70=buried (default), 60=medium, 50=shallow",
        )
        ric = st.number_input("Random initial conditions", value=2, min_value=1)

    job_name = st.text_input("Job name (optional)", value="MasterDock")

    st.markdown("---")
    submit_btn = st.button(
        "🚀 Submit to SwissDock",
        disabled=(receptor_file is None
                  or (ligand_file is None and not smiles_input)),
        use_container_width=True,
        type="primary",
    )

    if st.session_state.sd_session:
        st.markdown("---")
        st.markdown(f"**Session:** `{st.session_state.sd_session}`")
        poll_btn   = st.button("🔄 Check Status", use_container_width=True)
        cancel_btn = st.button("🛑 Cancel Job",   use_container_width=True)
        fetch_btn  = st.button("📥 Fetch Results", use_container_width=True)
    else:
        poll_btn = cancel_btn = fetch_btn = False


# ─────────────────────────────────────────────────────────────────────────────
#  Main content
# ─────────────────────────────────────────────────────────────────────────────
st.title("Molecular Interaction Analysis Platform")
st.markdown(
    "Upload your **Receptor PDB** and **Ligand SDF** (or SMILES) in the sidebar, "
    "configure the grid box, and click **🚀 Submit to SwissDock**."
)

# Server health
if not sd_check_server():
    st.warning(
        "⚠️ The SwissDock server (`swissdock.ch:8443`) could not be reached right now. "
        "The molecular viewer and file parsing will still work. "
        "Try submitting again in a few minutes."
    )

receptor_content = None
ligand_content   = None
ligand_mols      = []

# ── Receptor ─────────────────────────────────────────────────────────────────
if receptor_file is not None:
    st.markdown("---")
    st.markdown(
        f'<div class="section-header">🔵 Receptor: {receptor_file.name}</div>',
        unsafe_allow_html=True,
    )
    receptor_content = receptor_file.read().decode("utf-8")

    try:
        stats = parse_pdb(receptor_content)
        cols  = st.columns(4)
        for col, (k, v) in zip(cols, stats.items()):
            col.metric(k, v)
    except PDBConstructionException as e:
        st.error(f"PDB parse error: {e}")
        receptor_content = None

    if receptor_content:
        with st.expander("🔬 Receptor 3D Viewer", expanded=True):
            components.html(render_3d(receptor_content, "pdb", 420), height=440)

# ── Ligand ────────────────────────────────────────────────────────────────────
if ligand_file is not None:
    st.markdown("---")
    st.markdown(
        f'<div class="section-header">🟢 Ligand: {ligand_file.name}</div>',
        unsafe_allow_html=True,
    )
    ligand_content = ligand_file.read().decode("utf-8")
    try:
        ligand_mols = parse_sdf(ligand_content)
    except Exception as e:
        st.error(f"SDF parse error: {e}")

    if ligand_mols:
        st.write(f"Found **{len(ligand_mols)}** molecule(s).")
        st.dataframe(
            pd.DataFrame(
                [mol_props(m) for m in ligand_mols],
                index=[f"Mol {i+1}" for i in range(len(ligand_mols))],
            ),
            use_container_width=True,
        )
        with st.expander("🔬 Ligand 3D Viewer", expanded=True):
            components.html(render_3d(ligand_content, "sdf", 360), height=380)

        st.subheader("🖼 2D Structures")
        n = min(len(ligand_mols), 4)
        for col, mol in zip(st.columns(n), ligand_mols[:n]):
            img = render_2d(mol)
            if img:
                col.image(img, use_container_width=True)

elif smiles_input:
    st.markdown("---")
    st.markdown(
        '<div class="section-header">🟢 Ligand: SMILES input</div>',
        unsafe_allow_html=True,
    )
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        st.success("✅ Valid SMILES")
        st.json(mol_props(mol))
        img = render_2d(mol)
        if img:
            st.image(img, caption="Ligand 2D Structure", width=280)
    else:
        st.error("Invalid SMILES string. Please check and try again.")


# ─────────────────────────────────────────────────────────────────────────────
#  Submit to SwissDock
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="section-header">⚗️ SwissDock Docking</div>',
    unsafe_allow_html=True,
)

if submit_btn:
    st.session_state.docking_done   = False
    st.session_state.result_tarball = None

    log_area = st.empty()

    def log(msg: str):
        log_area.markdown(
            f'<div class="api-box">🔹 {msg}</div>', unsafe_allow_html=True
        )

    # ── Step 1: Prepare ligand ───────────────────────────────────────────────
    log("Step 1/5 — Uploading and preparing ligand on SwissDock…")

    mol2_bytes = None
    if ligand_file is not None and ligand_content:
        # Try to get a Mol2 from the SDF
        mol2_str = sdf_to_mol2_rdkit(ligand_content)
        if mol2_str:
            mol2_bytes = mol2_str.encode()
        else:
            # Fall back: derive SMILES from first molecule
            if ligand_mols:
                smiles_fallback = Chem.MolToSmiles(ligand_mols[0])
                ok, result = sd_prepare_ligand(
                    smiles=smiles_fallback,
                    use_vina=use_vina,
                )
            else:
                st.error("Cannot parse the SDF file into molecules.")
                st.stop()
    elif smiles_input:
        ok, result = sd_prepare_ligand(smiles=smiles_input, use_vina=use_vina)
    else:
        st.error("Please upload an SDF file or enter a SMILES string.")
        st.stop()

    if mol2_bytes:
        ok, result = sd_prepare_ligand(mol2_bytes=mol2_bytes, use_vina=use_vina)

    if not ok:
        st.error(f"❌ Ligand preparation failed:\n{result}")
        st.stop()

    session_num = result
    st.session_state.sd_session = session_num
    log(f"Step 1/5 — ✅ Ligand prepared. Session number: **{session_num}**")

    # ── Step 2: Prepare target ───────────────────────────────────────────────
    log("Step 2/5 — Uploading and preparing receptor on SwissDock…")

    if receptor_content is None:
        st.error("No receptor loaded. Please upload a PDB file.")
        st.stop()

    ok, msg = sd_prepare_target(
        pdb_bytes=receptor_content.encode(),
        session_number=session_num,
    )
    if not ok:
        st.error(f"❌ Target preparation failed:\n{msg}")
        st.stop()
    log("Step 2/5 — ✅ Receptor prepared.")

    # ── Step 3: Set parameters ────────────────────────────────────────────────
    log("Step 3/5 — Setting docking parameters…")
    can_submit, param_msg = sd_set_parameters(
        session_number=session_num,
        center_x=center_x, center_y=center_y, center_z=center_z,
        size_x=size_x, size_y=size_y, size_z=size_z,
        exhaust=int(exhaust),
        cavity=int(cavity),
        ric=int(ric),
        use_vina=use_vina,
        job_name=job_name or "MasterDock",
    )
    with st.expander("SwissDock parameter confirmation", expanded=False):
        st.code(param_msg)

    if not can_submit:
        st.error(
            "❌ SwissDock cannot submit this job. Common reasons:\n"
            "- No attractive cavity found in the search box → adjust Center X/Y/Z\n"
            "- Estimated calculation time exceeds server limit → reduce box size or exhaustiveness"
        )
        st.stop()
    log("Step 3/5 — ✅ Parameters accepted. Session can be submitted.")

    # ── Step 4: Start docking ─────────────────────────────────────────────────
    log("Step 4/5 — Submitting docking job to SwissDock queue…")
    ok, start_msg = sd_start_docking(session_num)
    if not ok:
        st.error(f"❌ Failed to start docking:\n{start_msg}")
        st.stop()
    log("Step 4/5 — ✅ Job submitted! Polling for results…")

    # ── Step 5: Poll until done ───────────────────────────────────────────────
    progress_bar = st.progress(0, text="Waiting in queue…")
    status_box   = st.empty()

    for poll_i in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        status = sd_check_status(session_num)
        status_box.info(f"⏳ Status: {status}")

        progress_bar.progress(
            min((poll_i + 1) / MAX_POLLS, 0.99),
            text=f"Poll {poll_i+1}: {status[:80]}",
        )

        if "finished" in status.lower():
            progress_bar.progress(1.0, text="✅ Docking finished!")
            log("Step 5/5 — ✅ Docking complete! Retrieving results…")

            tarball = sd_retrieve_results(session_num)
            if tarball:
                st.session_state.result_tarball = tarball
                st.session_state.docking_done   = True
            else:
                st.error("Could not retrieve results from SwissDock.")
            break

        if "error" in status.lower() or "failed" in status.lower():
            st.error(f"❌ SwissDock reported an error: {status}")
            break
    else:
        st.warning(
            "Job is taking longer than 60 minutes. "
            "Use **Check Status** and **Fetch Results** buttons once it finishes."
        )

# ── Manual controls (poll / cancel / fetch) ───────────────────────────────────
if poll_btn and st.session_state.sd_session:
    status = sd_check_status(st.session_state.sd_session)
    st.info(f"Status: {status}")

if cancel_btn and st.session_state.sd_session:
    msg = sd_cancel(st.session_state.sd_session)
    st.warning(f"Cancel response: {msg}")
    st.session_state.sd_session = None

if fetch_btn and st.session_state.sd_session:
    with st.spinner("Fetching results from SwissDock…"):
        tarball = sd_retrieve_results(st.session_state.sd_session)
    if tarball:
        st.session_state.result_tarball = tarball
        st.session_state.docking_done   = True
        st.success("Results fetched!")
    else:
        st.error("Could not fetch results. The job may not be finished yet.")


# ─────────────────────────────────────────────────────────────────────────────
#  Display Results
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.docking_done and st.session_state.result_tarball:
    st.markdown("---")
    st.markdown(
        '<div class="section-header">📊 Docking Results</div>',
        unsafe_allow_html=True,
    )

    parsed = parse_results_tarball(st.session_state.result_tarball)

    # Download button for raw results
    st.download_button(
        label="⬇️ Download full results (.tar.gz)",
        data=st.session_state.result_tarball,
        file_name=f"swissdock_{st.session_state.sd_session}.tar.gz",
        mime="application/gzip",
    )

    with st.expander("📁 Files in result archive"):
        st.write(parsed["files"])

    # Scores table
    if parsed["scores_df"] is not None:
        st.subheader("📈 Docking Scores")
        st.dataframe(parsed["scores_df"], use_container_width=True)
    elif parsed["log_text"]:
        st.subheader("📋 Score / Log Output")
        st.code(parsed["log_text"], language="text")
    else:
        st.info("No structured scores found in the results. Check the raw archive.")

    # 3-D viewer of best pose
    if parsed["poses_pdb"]:
        st.subheader("🔬 Docked Poses — 3D Viewer")
        pose_fmt = "pdbqt" if parsed["poses_pdb"].startswith("REMARK VINA") else "pdb"
        with st.expander("Best docked pose", expanded=True):
            components.html(
                render_3d(parsed["poses_pdb"], pose_fmt, 450), height=470
            )
        # 2-D of first pose (try parsing as SDF/PDB via RDKit)
        try:
            pose_mol = Chem.MolFromPDBBlock(parsed["poses_pdb"], sanitize=False)
            if pose_mol:
                img = render_2d(pose_mol)
                if img:
                    st.image(img, caption="Top pose (2D)", width=280)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Welcome / help screen
# ─────────────────────────────────────────────────────────────────────────────
if receptor_file is None and ligand_file is None and not smiles_input:
    st.info(
        "**👈 Upload your structures in the sidebar to get started.**\n\n"
        "**Docking workflow:**\n"
        "1. Upload **Receptor PDB** — any standard PDB file\n"
        "2. Upload **Ligand SDF** or enter a **SMILES** string\n"
        "3. Set the **Grid Box** center (X/Y/Z) and size (Å) to cover the binding site\n"
        "4. Choose **Attracting Cavities** (accurate) or **AutoDock Vina** (fast)\n"
        "5. Click **🚀 Submit to SwissDock** — the job runs on SIB's servers\n"
        "6. Results appear automatically when the job finishes (or use **Fetch Results**)\n\n"
        "> **No local installation needed.** All docking runs on the SwissDock server "
        "at the Swiss Institute of Bioinformatics. Free for academic use.\n\n"
        "> **Note on Mol2 format:** SwissDock's AC engine prefers Mol2 ligand files. "
        "If OpenBabel (`obabel`) is not installed, the app falls back to SMILES submission."
    )

st.markdown("---")
st.caption(
    "MasterDock · Docking: **SwissDock REST API** (SIB, UNIL) · "
    "py3Dmol · RDKit · Biopython · "
    "[SwissDock 2024 paper](https://academic.oup.com/nar/article/52/W1/W324/7660078)"
)
