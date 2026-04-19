# -*- coding: utf-8 -*-
"""
MasterDock – Molecular Interaction Analysis Platform
Docking engine : SwissDock REST API (https://swissdock.ch:8443)
Visualization  : 3Dmol.js via CDN (3D) + RDKit (2D)
Parsing        : Biopython (PDB) + RDKit (SDF)
"""

import io
import os
import re
import tarfile
import tempfile
import time

import pandas as pd
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
SD_BASE        = "https://swissdock.ch:8443"
SD_HELLO       = f"{SD_BASE}/"
SD_PREPLIG     = f"{SD_BASE}/preplig"
SD_PREPTARGET  = f"{SD_BASE}/preptarget"
SD_SETPARAMS   = f"{SD_BASE}/setparameters"
SD_STARTDOCK   = f"{SD_BASE}/startdock"
SD_CHECKSTATUS = f"{SD_BASE}/checkstatus"
SD_RETRIEVE    = f"{SD_BASE}/retrievesession"
SD_CANCEL      = f"{SD_BASE}/cancelsession"

POLL_INTERVAL = 15    # seconds between status polls
MAX_POLLS     = 240   # 240 × 15 s = 60 min max wait

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
#  3D Viewer — uses 3Dmol.js via CDN (no IPython required)
# ─────────────────────────────────────────────────────────────────────────────

def render_3d(content: str, fmt: str, height: int = 420) -> str:
    """
    Return a self-contained HTML page that renders a 3Dmol.js viewer.

    KEY FIX: py3Dmol._repr_html_() raises
        ImportError: This function requires an active IPython notebook.
    when used outside Jupyter. We bypass py3Dmol entirely and call the
    underlying 3Dmol.js library directly via CDN, which works perfectly
    inside Streamlit's components.html().
    """
    # Escape the molecule data so it is safe inside a JS string.
    # Replace backslashes first, then single quotes.
    safe = content.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    return f"""<!DOCTYPE html>
<html>
<head><style>
  body {{ margin:0; padding:0; background:#1e1e2e; }}
  #viewer {{ width:100%; height:{height}px; position:relative; }}
</style></head>
<body>
  <div id="viewer"></div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.3/3Dmol-min.js"></script>
  <script>
    var config = {{ backgroundColor: 0x1e1e2e }};
    var viewer = $3Dmol.createViewer(document.getElementById('viewer'), config);
    var data = '{safe}';
    viewer.addModel(data, '{fmt.lower()}');
    viewer.setStyle({{}}, {{ stick: {{}} }});
    viewer.zoomTo();
    viewer.render();
  </script>
</body>
</html>"""


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


def render_2d(mol, size=(280, 280)):
    return Draw.MolToImage(mol, size=size) if mol else None


def mol_props(mol) -> dict:
    return {
        "Atoms": mol.GetNumAtoms(),
        "Bonds": mol.GetNumBonds(),
        "MW (Da)": round(rdMolDescriptors.CalcExactMolWt(mol), 3),
    }


def sdf_to_mol2_via_obabel(sdf_content: str):
    """Convert first SDF molecule to Mol2 via obabel. Returns bytes or None."""
    mols = parse_sdf(sdf_content)
    if not mols:
        return None
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sdf", delete=False) as f:
        f.write(sdf_content)
        sdf_path = f.name
    mol2_path = sdf_path.replace(".sdf", ".mol2")
    try:
        import subprocess
        subprocess.run(
            ["obabel", sdf_path, "-O", mol2_path],
            capture_output=True, check=True,
        )
        if os.path.exists(mol2_path):
            with open(mol2_path, "rb") as f:
                return f.read()
        return None
    except Exception:
        return None
    finally:
        for p in (sdf_path, mol2_path):
            if os.path.exists(p):
                os.remove(p)


# ─────────────────────────────────────────────────────────────────────────────
#  SwissDock REST API client
# ─────────────────────────────────────────────────────────────────────────────

def sd_check_server() -> bool:
    try:
        r = requests.get(SD_HELLO, timeout=10, verify=False)
        return "Hello World" in r.text
    except Exception:
        return False


def sd_prepare_ligand(mol2_bytes=None, smiles=None, session_number=None, use_vina=False):
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
            return False, "No ligand input provided."
        text = r.text.strip()
        m = re.search(r"Session number:\s*(\d+)", text)
        if m:
            return True, m.group(1)
        m2 = re.search(r"Using session number:\s*(\d+)", text)
        if m2:
            return True, m2.group(1)
        return False, f"Unexpected SwissDock response:\n{text}"
    except Exception as e:
        return False, f"Network error preparing ligand: {e}"


def sd_prepare_target(pdb_bytes: bytes, session_number: str):
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


def sd_set_parameters(session_number, center_x, center_y, center_z,
                      size_x=20.0, size_y=20.0, size_z=20.0,
                      exhaust=90, cavity=70, ric=2, use_vina=False, job_name="MasterDock"):
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


def sd_start_docking(session_number: str):
    try:
        r = requests.get(SD_STARTDOCK, params={"sessionNumber": session_number}, timeout=60, verify=False)
        text = r.text.strip()
        return "submitted" in text.lower(), text
    except Exception as e:
        return False, f"Network error starting docking: {e}"


def sd_check_status(session_number: str) -> str:
    try:
        r = requests.get(SD_CHECKSTATUS, params={"sessionNumber": session_number}, timeout=30, verify=False)
        return r.text.strip()
    except Exception as e:
        return f"Error checking status: {e}"


def sd_retrieve_results(session_number: str):
    try:
        r = requests.get(SD_RETRIEVE, params={"sessionNumber": session_number},
                         timeout=300, verify=False)
        if r.status_code == 200 and len(r.content) > 100:
            return r.content
        return None
    except Exception:
        return None


def sd_cancel(session_number: str) -> str:
    try:
        r = requests.get(SD_CANCEL, params={"sessionNumber": session_number},
                         timeout=30, verify=False)
        return r.text.strip()
    except Exception as e:
        return str(e)


def parse_results_tarball(tar_bytes: bytes) -> dict:
    result = {"poses_pdb": None, "scores_df": None, "log_text": None, "files": []}
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            members = tar.getmembers()
            result["files"] = [m.name for m in members]
            for member in members:
                f = tar.extractfile(member)
                if f is None:
                    continue
                raw  = f.read()
                name = member.name.lower()
                if name.endswith(".pdb") and "cluster" in name:
                    result["poses_pdb"] = raw.decode("utf-8", errors="ignore")
                elif name.endswith(".pdbqt") and "out" in name:
                    result["poses_pdb"] = raw.decode("utf-8", errors="ignore")
                elif name.endswith(".log") or "score" in name or "cluster" in name:
                    result["log_text"] = raw.decode("utf-8", errors="ignore")

        if result["log_text"]:
            scores = []
            for line in result["log_text"].splitlines():
                m = re.match(r"\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
                if m:
                    scores.append({
                        "Mode": int(m.group(1)),
                        "Affinity (kcal/mol)": float(m.group(2)),
                        "RMSD lb (Å)": float(m.group(3)),
                        "RMSD ub (Å)": float(m.group(4)),
                    })
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
    st.caption("Powered by the **SwissDock REST API** · 3Dmol.js · RDKit · Biopython")
    st.markdown("---")

    st.markdown("### 📂 Upload Structures")
    receptor_file = st.file_uploader("Receptor PDB", type=["pdb"], key="rec_pdb")

    lig_input_mode = st.radio("Ligand input", ["Upload SDF file", "Enter SMILES"], horizontal=True)
    if lig_input_mode == "Upload SDF file":
        ligand_file  = st.file_uploader("Ligand SDF", type=["sdf"], key="lig_sdf")
        smiles_input = None
    else:
        ligand_file  = None
        smiles_input = st.text_input("Ligand SMILES", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")

    st.markdown("---")
    st.markdown("### 🔬 Docking Engine")
    engine   = st.radio("Algorithm", ["Attracting Cavities 2.0 (AC)", "AutoDock Vina"])
    use_vina = (engine == "AutoDock Vina")

    st.markdown("---")
    st.markdown("### ⚙️ Grid Box (Å)")
    c1, c2, c3 = st.columns(3)
    center_x = c1.number_input("X", value=0.0, format="%.1f")
    center_y = c2.number_input("Y", value=0.0, format="%.1f")
    center_z = c3.number_input("Z", value=0.0, format="%.1f")
    c1, c2, c3 = st.columns(3)
    size_x = c1.number_input("SX", value=20.0, format="%.1f")
    size_y = c2.number_input("SY", value=20.0, format="%.1f")
    size_z = c3.number_input("SZ", value=20.0, format="%.1f")

    st.markdown("### 🎛️ Sampling")
    if use_vina:
        exhaust       = st.slider("Exhaustiveness", 1, 64, 8)
        cavity, ric   = 70, 2
    else:
        exhaust = st.select_slider("Exhaustivity (°)", options=[180, 90, 60], value=90)
        cavity  = st.select_slider("Cavity prioritization", options=[50, 60, 70], value=70)
        ric     = st.number_input("Random initial conditions", value=2, min_value=1)

    job_name = st.text_input("Job name", value="MasterDock")

    st.markdown("---")
    submit_btn = st.button(
        "🚀 Submit to SwissDock",
        disabled=(receptor_file is None and True
                  if (ligand_file is None and not smiles_input) else
                  receptor_file is None),
        use_container_width=True,
        type="primary",
    )

    if st.session_state.sd_session:
        st.markdown("---")
        st.markdown(f"**Session:** `{st.session_state.sd_session}`")
        poll_btn   = st.button("🔄 Check Status",  use_container_width=True)
        cancel_btn = st.button("🛑 Cancel Job",     use_container_width=True)
        fetch_btn  = st.button("📥 Fetch Results",  use_container_width=True)
    else:
        poll_btn = cancel_btn = fetch_btn = False


# ─────────────────────────────────────────────────────────────────────────────
#  Main content
# ─────────────────────────────────────────────────────────────────────────────
st.title("Molecular Interaction Analysis Platform")
st.markdown(
    "Upload your **Receptor PDB** and **Ligand SDF** (or SMILES) in the sidebar, "
    "set the grid box, then click **🚀 Submit to SwissDock**."
)

# Server ping
if not sd_check_server():
    st.warning(
        "⚠️ SwissDock server (`swissdock.ch:8443`) is unreachable right now. "
        "Visualization still works. Try submitting in a few minutes."
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
    st.markdown('<div class="section-header">🟢 Ligand: SMILES input</div>', unsafe_allow_html=True)
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
#  SwissDock submission
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">⚗️ SwissDock Docking</div>', unsafe_allow_html=True)

if submit_btn:
    st.session_state.docking_done   = False
    st.session_state.result_tarball = None
    log_area = st.empty()

    def log(msg):
        log_area.markdown(f'<div class="api-box">🔹 {msg}</div>', unsafe_allow_html=True)

    # Step 1 — prepare ligand
    log("Step 1/5 — Uploading ligand to SwissDock…")
    mol2_bytes = None
    ok = False

    if ligand_file is not None and ligand_content:
        mol2_bytes = sdf_to_mol2_via_obabel(ligand_content)
        if mol2_bytes:
            ok, result = sd_prepare_ligand(mol2_bytes=mol2_bytes, use_vina=use_vina)
        else:
            # fallback: SMILES derived from first molecule
            if ligand_mols:
                fb_smiles = Chem.MolToSmiles(ligand_mols[0])
                ok, result = sd_prepare_ligand(smiles=fb_smiles, use_vina=use_vina)
            else:
                st.error("Cannot parse the SDF file.")
                st.stop()
    elif smiles_input:
        ok, result = sd_prepare_ligand(smiles=smiles_input, use_vina=use_vina)
    else:
        st.error("Please upload an SDF file or enter a SMILES string.")
        st.stop()

    if not ok:
        st.error(f"❌ Ligand preparation failed:\n{result}")
        st.stop()

    session_num = result
    st.session_state.sd_session = session_num
    log(f"Step 1/5 — ✅ Ligand prepared. Session: **{session_num}**")

    # Step 2 — prepare target
    log("Step 2/5 — Uploading receptor to SwissDock…")
    if receptor_content is None:
        st.error("No receptor loaded. Please upload a PDB file.")
        st.stop()

    ok, msg = sd_prepare_target(receptor_content.encode(), session_num)
    if not ok:
        st.error(f"❌ Target preparation failed:\n{msg}")
        st.stop()
    log("Step 2/5 — ✅ Receptor prepared.")

    # Step 3 — set parameters
    log("Step 3/5 — Setting docking parameters…")
    can_submit, param_msg = sd_set_parameters(
        session_num,
        center_x, center_y, center_z,
        size_x, size_y, size_z,
        int(exhaust), int(cavity), int(ric),
        use_vina, job_name or "MasterDock",
    )
    with st.expander("SwissDock parameter confirmation"):
        st.code(param_msg)

    if not can_submit:
        st.error(
            "❌ SwissDock cannot submit this job.\n"
            "- No cavity found → adjust Center X/Y/Z to the binding site\n"
            "- Calculation too long → reduce box size or exhaustiveness"
        )
        st.stop()
    log("Step 3/5 — ✅ Parameters accepted.")

    # Step 4 — start docking
    log("Step 4/5 — Submitting to SwissDock queue…")
    ok, start_msg = sd_start_docking(session_num)
    if not ok:
        st.error(f"❌ Failed to start docking:\n{start_msg}")
        st.stop()
    log("Step 4/5 — ✅ Job submitted. Polling for results…")

    # Step 5 — poll
    progress_bar = st.progress(0, text="Waiting in queue…")
    status_box   = st.empty()

    for poll_i in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        status = sd_check_status(session_num)
        status_box.info(f"⏳ {status}")
        progress_bar.progress(min((poll_i + 1) / MAX_POLLS, 0.99),
                              text=f"Poll {poll_i+1}: {status[:80]}")

        if "finished" in status.lower():
            progress_bar.progress(1.0, text="✅ Done!")
            log("Step 5/5 — ✅ Docking complete! Retrieving results…")
            tarball = sd_retrieve_results(session_num)
            if tarball:
                st.session_state.result_tarball = tarball
                st.session_state.docking_done   = True
            else:
                st.error("Could not retrieve results from SwissDock.")
            break

        if "error" in status.lower() or "failed" in status.lower():
            st.error(f"❌ SwissDock error: {status}")
            break
    else:
        st.warning(
            "Job taking longer than 60 min. Use **Check Status** and "
            "**Fetch Results** buttons in the sidebar once it finishes."
        )

# Manual controls
if poll_btn and st.session_state.sd_session:
    st.info(f"Status: {sd_check_status(st.session_state.sd_session)}")

if cancel_btn and st.session_state.sd_session:
    st.warning(sd_cancel(st.session_state.sd_session))
    st.session_state.sd_session = None

if fetch_btn and st.session_state.sd_session:
    with st.spinner("Fetching results…"):
        tarball = sd_retrieve_results(st.session_state.sd_session)
    if tarball:
        st.session_state.result_tarball = tarball
        st.session_state.docking_done   = True
        st.success("Results fetched!")
    else:
        st.error("Not ready yet — try again in a moment.")


# ─────────────────────────────────────────────────────────────────────────────
#  Results
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.docking_done and st.session_state.result_tarball:
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Docking Results</div>', unsafe_allow_html=True)
    parsed = parse_results_tarball(st.session_state.result_tarball)

    st.download_button(
        "⬇️ Download full results (.tar.gz)",
        data=st.session_state.result_tarball,
        file_name=f"swissdock_{st.session_state.sd_session}.tar.gz",
        mime="application/gzip",
    )

    with st.expander("📁 Files in archive"):
        st.write(parsed["files"])

    if parsed["scores_df"] is not None:
        st.subheader("📈 Docking Scores")
        st.dataframe(parsed["scores_df"], use_container_width=True)
    elif parsed["log_text"]:
        st.subheader("📋 Log Output")
        st.code(parsed["log_text"])

    if parsed["poses_pdb"]:
        pose_fmt = "pdbqt" if "REMARK VINA" in parsed["poses_pdb"] else "pdb"
        st.subheader("🔬 Docked Poses — 3D Viewer")
        with st.expander("Best docked pose", expanded=True):
            components.html(render_3d(parsed["poses_pdb"], pose_fmt, 450), height=470)


# ─────────────────────────────────────────────────────────────────────────────
#  Welcome screen
# ─────────────────────────────────────────────────────────────────────────────
if receptor_file is None and ligand_file is None and not smiles_input:
    st.info(
        "**👈 Upload your structures in the sidebar to get started.**\n\n"
        "**Workflow:**\n"
        "1. Upload **Receptor PDB** → 3D viewer + stats\n"
        "2. Upload **Ligand SDF** or type a **SMILES** string → 3D/2D viewer\n"
        "3. Set **Grid Box** center (X/Y/Z) and size over the binding site\n"
        "4. Choose **Attracting Cavities** (accurate) or **Vina** (fast)\n"
        "5. Click **🚀 Submit to SwissDock** — docking runs on SIB's servers\n"
        "6. Results appear automatically (or use **Fetch Results** for long jobs)\n\n"
        "> Free for academic use · No local software installation required"
    )

st.markdown("---")
st.caption(
    "MasterDock · Docking: SwissDock REST API (SIB / UNIL) · "
    "3D: 3Dmol.js · Cheminformatics: RDKit · Parsing: Biopython"
)
