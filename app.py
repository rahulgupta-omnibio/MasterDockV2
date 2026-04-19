# -*- coding: utf-8 -*-
"""
MasterDock – Molecular Interaction Analysis Platform
Docking : SwissDock REST API  https://swissdock.ch:8443
3D view : 3Dmol.js via CDN  (no IPython needed)
2D view : RDKit
Parsing : Biopython (PDB) + RDKit (SDF)
"""

import io
import os
import re
import tarfile
import tempfile
import time
import warnings

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors

warnings.filterwarnings("ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning)

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MasterDock · SwissDock", page_icon="🧬", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
#  SwissDock API endpoints
# ─────────────────────────────────────────────────────────────────────────────
SD_BASE        = "https://swissdock.ch:8443"
SD_PREPLIG     = f"{SD_BASE}/preplig"
SD_PREPTARGET  = f"{SD_BASE}/preptarget"
SD_SETPARAMS   = f"{SD_BASE}/setparameters"
SD_STARTDOCK   = f"{SD_BASE}/startdock"
SD_CHECKSTATUS = f"{SD_BASE}/checkstatus"
SD_RETRIEVE    = f"{SD_BASE}/retrievesession"
SD_CANCEL      = f"{SD_BASE}/cancelsession"

POLL_INTERVAL = 15
MAX_POLLS     = 240

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container{padding-top:1.5rem}
.sec-hdr{background:linear-gradient(90deg,#1a237e,#283593);color:#fff;
  padding:8px 16px;border-radius:6px;font-size:1.05rem;font-weight:600;margin-bottom:10px}
.tip-box{background:#e8f5e9;border-left:4px solid #2e7d32;padding:10px 14px;
  border-radius:4px;font-size:.85rem;margin-bottom:8px;color:#1b5e20}
.warn-box{background:#fff3e0;border-left:4px solid #e65100;padding:10px 14px;
  border-radius:4px;font-size:.85rem;margin-bottom:8px;color:#bf360c}
.api-log{background:#f0f4ff;border-left:4px solid #3949ab;padding:8px 14px;
  border-radius:4px;font-size:.83rem;margin-bottom:6px}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  3D viewer — 3Dmol.js CDN (no IPython / no py3Dmol needed)
# ─────────────────────────────────────────────────────────────────────────────
def render_3d(content: str, fmt: str, height: int = 420) -> str:
    safe = content.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    return f"""<!DOCTYPE html><html><head>
<style>body{{margin:0;padding:0;background:#1e1e2e}}
#v{{width:100%;height:{height}px}}</style></head><body>
<div id="v"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.3/3Dmol-min.js"></script>
<script>
var viewer=$3Dmol.createViewer(document.getElementById('v'),{{backgroundColor:0x1e1e2e}});
viewer.addModel('{safe}','{fmt.lower()}');
viewer.setStyle({{}},{{stick:{{}}}});
viewer.zoomTo();viewer.render();
</script></body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Molecular helpers
# ─────────────────────────────────────────────────────────────────────────────
def parse_pdb(txt: str) -> dict:
    p = PDBParser(QUIET=True)
    try:
        s = p.get_structure("m", io.StringIO(txt))
    except Exception as e:
        raise PDBConstructionException(str(e)) from e
    d = {"Models": 0, "Chains": 0, "Residues": 0, "Atoms": 0}
    for model in s:
        d["Models"] += 1
        for chain in model:
            d["Chains"] += 1
            for res in chain:
                d["Residues"] += 1
                for _ in res:
                    d["Atoms"] += 1
    return d


def parse_sdf(txt: str):
    sup = Chem.SDMolSupplier()
    sup.SetData(txt, removeHs=False)
    return [m for m in sup if m is not None]


def render_2d(mol, size=(280, 280)):
    return Draw.MolToImage(mol, size=size) if mol else None


def mol_props(mol) -> dict:
    return {
        "Atoms": mol.GetNumAtoms(),
        "Bonds": mol.GetNumBonds(),
        "MW (Da)": round(rdMolDescriptors.CalcExactMolWt(mol), 3),
    }


def sdf_to_mol2(sdf_txt: str):
    """Convert first molecule in SDF to Mol2 bytes via obabel."""
    mols = parse_sdf(sdf_txt)
    if not mols:
        return None
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sdf", delete=False) as f:
        f.write(sdf_txt); sdf_p = f.name
    mol2_p = sdf_p.replace(".sdf", ".mol2")
    try:
        import subprocess
        subprocess.run(["obabel", sdf_p, "-O", mol2_p], capture_output=True, check=True)
        if os.path.exists(mol2_p):
            return open(mol2_p, "rb").read()
    except Exception:
        pass
    finally:
        for p in (sdf_p, mol2_p):
            if os.path.exists(p): os.remove(p)
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  SwissDock API calls
# ─────────────────────────────────────────────────────────────────────────────
def _post(url, **kw):
    return requests.post(url, timeout=180, verify=False, **kw)

def _get(url, **kw):
    return requests.get(url, timeout=60, verify=False, **kw)


def sd_alive() -> bool:
    try:
        return "Hello" in requests.get(f"{SD_BASE}/", timeout=8, verify=False).text
    except Exception:
        return False


def sd_prep_ligand(mol2_bytes=None, smiles=None, session=None, vina=False):
    params = {}
    if session: params["sessionNumber"] = session
    if vina:    params["Vina"] = ""
    try:
        if mol2_bytes:
            r = _post(SD_PREPLIG, files={"myLig": ("lig.mol2", mol2_bytes, "chemical/x-mol2")}, params=params)
        elif smiles:
            params["mySMILES"] = smiles
            r = _get(SD_PREPLIG, params=params)
        else:
            return False, "No ligand input."
        txt = r.text.strip()
        m = re.search(r"[Ss]ession number:\s*(\d+)", txt)
        return (True, m.group(1)) if m else (False, f"SwissDock said:\n{txt}")
    except Exception as e:
        return False, str(e)


def sd_prep_target(pdb_bytes: bytes, session: str):
    try:
        r = _post(SD_PREPTARGET,
                  files={"myTarget": ("target.pdb", pdb_bytes, "chemical/x-pdb")},
                  params={"sessionNumber": session})
        txt = r.text.strip()
        ok = "prepared" in txt.lower()
        return ok, txt
    except Exception as e:
        return False, str(e)


def sd_set_params(session, cx, cy, cz, sx, sy, sz,
                  exhaust, cavity, ric, vina, name, blind):
    params = {"sessionNumber": session, "exhaust": exhaust, "name": name}
    if not blind:
        params["boxCenter"] = f"{cx}_{cy}_{cz}"
        params["boxSize"]   = f"{sx}_{sy}_{sz}"
    if not vina:
        params["cavity"] = cavity
        params["ric"]    = ric
    try:
        r = _get(SD_SETPARAMS, params=params)
        txt = r.text.strip()
        ok  = "session can be submitted" in txt.lower()
        return ok, txt
    except Exception as e:
        return False, str(e)


def sd_start(session: str):
    try:
        r = _get(SD_STARTDOCK, params={"sessionNumber": session})
        txt = r.text.strip()
        return "submitted" in txt.lower(), txt
    except Exception as e:
        return False, str(e)


def sd_status(session: str) -> str:
    try:
        return _get(SD_CHECKSTATUS, params={"sessionNumber": session}).text.strip()
    except Exception as e:
        return str(e)


def sd_retrieve(session: str):
    try:
        r = requests.get(SD_RETRIEVE, params={"sessionNumber": session},
                         timeout=300, verify=False)
        return r.content if r.status_code == 200 and len(r.content) > 100 else None
    except Exception:
        return None


def sd_cancel(session: str) -> str:
    try:
        return _get(SD_CANCEL, params={"sessionNumber": session}).text.strip()
    except Exception as e:
        return str(e)


def parse_tarball(tar_bytes: bytes) -> dict:
    res = {"poses": None, "scores": None, "log": None, "files": []}
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            res["files"] = [m.name for m in tar.getmembers()]
            for member in tar.getmembers():
                f = tar.extractfile(member)
                if not f: continue
                raw  = f.read()
                name = member.name.lower()
                if (name.endswith(".pdb") and "cluster" in name) or \
                   (name.endswith(".pdbqt") and "out" in name):
                    res["poses"] = raw.decode("utf-8", errors="ignore")
                if name.endswith(".log") or "score" in name or \
                   ("cluster" in name and name.endswith(".txt")):
                    res["log"] = raw.decode("utf-8", errors="ignore")
        if res["log"]:
            rows = []
            for line in res["log"].splitlines():
                m = re.match(r"\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
                if m:
                    rows.append({"Mode": int(m.group(1)),
                                 "Affinity (kcal/mol)": float(m.group(2)),
                                 "RMSD lb (Å)": float(m.group(3)),
                                 "RMSD ub (Å)": float(m.group(4))})
                m2 = re.match(r"\s*(\d+)\s+([-\d.]+)\s*$", line.strip())
                if m2:
                    rows.append({"Cluster": int(m2.group(1)),
                                 "Energy (kcal/mol)": float(m2.group(2))})
            if rows: res["scores"] = pd.DataFrame(rows)
    except Exception as e:
        res["log"] = f"Parse error: {e}"
    return res


# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────
for k, v in {"sd_session": None, "done": False, "tarball": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 MasterDock")
    st.caption("SwissDock API · 3Dmol.js · RDKit · Biopython")
    st.markdown("---")

    # ── Structures
    st.markdown("### 📂 Upload Structures")
    receptor_file = st.file_uploader("Receptor PDB", type=["pdb"], key="rec")
    lig_mode = st.radio("Ligand input", ["Upload SDF", "Enter SMILES"], horizontal=True)
    if lig_mode == "Upload SDF":
        ligand_file  = st.file_uploader("Ligand SDF", type=["sdf"], key="lig")
        smiles_input = None
    else:
        ligand_file  = None
        smiles_input = st.text_input("SMILES", placeholder="CC(=O)Oc1ccccc1C(=O)O")

    # ── Engine
    st.markdown("---")
    st.markdown("### 🔬 Docking Engine")
    engine   = st.radio("Algorithm", ["Attracting Cavities 2.0 (AC)", "AutoDock Vina"])
    use_vina = engine == "AutoDock Vina"

    # ── Grid box
    st.markdown("---")
    st.markdown("### ⚙️ Grid Box")

    blind = st.checkbox(
        "Blind docking (no box needed)",
        value=True,
        help="SwissDock auto-detects all cavities on the protein surface. "
             "Recommended if you don't know the exact binding site. "
             "Only works with Attracting Cavities engine.",
    )

    if blind and use_vina:
        st.warning("Blind docking requires AC engine. Switching effect: Vina needs a box.")

    if not blind:
        st.markdown("**Box center (Å)**")
        c1, c2, c3 = st.columns(3)
        center_x = c1.number_input("X", value=0.0, format="%.1f", key="cx")
        center_y = c2.number_input("Y", value=0.0, format="%.1f", key="cy")
        center_z = c3.number_input("Z", value=0.0, format="%.1f", key="cz")

        st.markdown("**Box size (Å)**")
        c1, c2, c3 = st.columns(3)
        size_x = c1.number_input("SX", value=20.0, format="%.1f", key="sx")
        size_y = c2.number_input("SY", value=20.0, format="%.1f", key="sy")
        size_z = c3.number_input("SZ", value=20.0, format="%.1f", key="sz")

        if use_vina:
            vol = size_x * size_y * size_z
            if vol > 27000:  # 30^3
                st.markdown(
                    '<div class="warn-box">⚠️ Box too large for Vina (10 min limit).<br>'
                    'Keep each dimension ≤ 30 Å for Vina.<br>'
                    'Use AC engine for larger search spaces.</div>',
                    unsafe_allow_html=True,
                )
    else:
        center_x = center_y = center_z = 0.0
        size_x   = size_y   = size_z   = 20.0

    # ── Sampling
    st.markdown("### 🎛️ Sampling")
    if use_vina:
        exhaust     = st.slider("Exhaustiveness (1–8 for Vina)", 1, 8, 4,
                                help="Keep ≤ 4 to stay within SwissDock's 10-min Vina limit.")
        cavity = ric = 70
    else:
        exhaust = st.select_slider("Exhaustivity (°)", [180, 90, 60], 90,
                                   help="180=fast, 90=default, 60=thorough")
        cavity  = st.select_slider("Cavity depth", [50, 60, 70], 70,
                                   help="70=deep buried, 50=shallow surface")
        ric     = st.number_input("Random initial conditions", 1, 10, 2)

    job_name = st.text_input("Job name", "MasterDock")

    st.markdown("---")
    can_submit_btn = (
        receptor_file is not None and
        (ligand_file is not None or (smiles_input and smiles_input.strip()))
    )
    submit_btn = st.button(
        "🚀 Submit to SwissDock",
        disabled=not can_submit_btn,
        use_container_width=True,
        type="primary",
    )

    if st.session_state.sd_session:
        st.markdown("---")
        st.info(f"Session: `{st.session_state.sd_session}`")
        poll_btn   = st.button("🔄 Check Status",  use_container_width=True)
        cancel_btn = st.button("🛑 Cancel Job",     use_container_width=True)
        fetch_btn  = st.button("📥 Fetch Results",  use_container_width=True)
    else:
        poll_btn = cancel_btn = fetch_btn = False


# ─────────────────────────────────────────────────────────────────────────────
#  Main content
# ─────────────────────────────────────────────────────────────────────────────
st.title("Molecular Interaction Analysis Platform")

# Server status
alive = sd_alive()
if not alive:
    st.warning("⚠️ SwissDock server unreachable right now. Visualization still works.")

receptor_content = ligand_content = None
ligand_mols = []

# ── Receptor
if receptor_file is not None:
    st.markdown("---")
    st.markdown(f'<div class="sec-hdr">🔵 Receptor: {receptor_file.name}</div>',
                unsafe_allow_html=True)
    receptor_content = receptor_file.read().decode("utf-8")
    try:
        stats = parse_pdb(receptor_content)
        cols  = st.columns(4)
        for col, (k, v) in zip(cols, stats.items()):
            col.metric(k, v)
        with st.expander("🔬 Receptor 3D Viewer", expanded=True):
            components.html(render_3d(receptor_content, "pdb", 420), height=440)
    except PDBConstructionException as e:
        st.error(f"PDB parse error: {e}")
        receptor_content = None

# ── Ligand
if ligand_file is not None:
    st.markdown("---")
    st.markdown(f'<div class="sec-hdr">🟢 Ligand: {ligand_file.name}</div>',
                unsafe_allow_html=True)
    ligand_content = ligand_file.read().decode("utf-8")
    try:
        ligand_mols = parse_sdf(ligand_content)
    except Exception as e:
        st.error(f"SDF parse error: {e}")

    if ligand_mols:
        st.write(f"Found **{len(ligand_mols)}** molecule(s).")
        st.dataframe(pd.DataFrame([mol_props(m) for m in ligand_mols],
                                  index=[f"Mol {i+1}" for i in range(len(ligand_mols))]),
                     use_container_width=True)
        with st.expander("🔬 Ligand 3D Viewer", expanded=True):
            components.html(render_3d(ligand_content, "sdf", 360), height=380)
        st.subheader("🖼 2D Structures")
        n = min(len(ligand_mols), 4)
        for col, mol in zip(st.columns(n), ligand_mols[:n]):
            img = render_2d(mol)
            if img:
                col.image(img, use_container_width=True)

elif smiles_input and smiles_input.strip():
    st.markdown("---")
    st.markdown('<div class="sec-hdr">🟢 Ligand: SMILES</div>', unsafe_allow_html=True)
    mol = Chem.MolFromSmiles(smiles_input.strip())
    if mol:
        st.success("✅ Valid SMILES")
        st.json(mol_props(mol))
        img = render_2d(mol)
        if img:
            st.image(img, width=280)
    else:
        st.error("Invalid SMILES — please check the string.")


# ─────────────────────────────────────────────────────────────────────────────
#  Docking section
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="sec-hdr">⚗️ SwissDock Docking</div>', unsafe_allow_html=True)

# Tips
if not blind:
    st.markdown("""
<div class="tip-box">
<b>Box tips</b><br>
• Center X/Y/Z = coordinates of the binding site from your PDB file or literature<br>
• For Vina: keep each dimension ≤ 30 Å and exhaustiveness ≤ 4<br>
• For AC: box can be larger; or enable <b>Blind docking</b> to skip box entirely
</div>""", unsafe_allow_html=True)
else:
    st.markdown("""
<div class="tip-box">
<b>Blind docking mode</b> — SwissDock will scan ALL cavities on the protein automatically.
No box needed. Works with Attracting Cavities engine only.
Results may take longer but require zero knowledge of the binding site.
</div>""", unsafe_allow_html=True)


if submit_btn:
    st.session_state.done    = False
    st.session_state.tarball = None
    log_slot = st.empty()

    def log(msg):
        log_slot.markdown(f'<div class="api-log">🔹 {msg}</div>', unsafe_allow_html=True)

    # ── Step 1: ligand
    log("Step 1/5 — Preparing ligand on SwissDock…")
    effective_vina = use_vina and not blind

    mol2_bytes = None
    if ligand_file and ligand_content:
        mol2_bytes = sdf_to_mol2(ligand_content)
        if mol2_bytes:
            ok, result = sd_prep_ligand(mol2_bytes=mol2_bytes, vina=effective_vina)
        else:
            fb = Chem.MolToSmiles(ligand_mols[0]) if ligand_mols else None
            if not fb:
                st.error("Cannot parse SDF."); st.stop()
            ok, result = sd_prep_ligand(smiles=fb, vina=effective_vina)
    elif smiles_input and smiles_input.strip():
        ok, result = sd_prep_ligand(smiles=smiles_input.strip(), vina=effective_vina)
    else:
        st.error("No ligand provided."); st.stop()

    if not ok:
        st.error(f"❌ Ligand prep failed. SwissDock said:\n\n```\n{result}\n```")
        st.stop()

    session_id = result
    st.session_state.sd_session = session_id
    log(f"Step 1/5 — ✅ Ligand ready. Session: **{session_id}**")

    # ── Step 2: receptor
    log("Step 2/5 — Preparing receptor on SwissDock…")
    if not receptor_content:
        st.error("No receptor PDB loaded."); st.stop()
    ok, msg = sd_prep_target(receptor_content.encode(), session_id)
    if not ok:
        st.error(f"❌ Target prep failed. SwissDock said:\n\n```\n{msg}\n```")
        st.stop()
    log("Step 2/5 — ✅ Receptor ready.")

    # ── Step 3: parameters
    log("Step 3/5 — Setting parameters…")
    ok, param_msg = sd_set_params(
        session_id,
        center_x, center_y, center_z,
        size_x, size_y, size_z,
        int(exhaust), int(cavity), int(ric),
        effective_vina, job_name or "MasterDock",
        blind=blind,
    )

    # Always show the raw SwissDock response
    with st.expander("📋 SwissDock parameter response (click to read)", expanded=not ok):
        st.code(param_msg, language="text")

    if not ok:
        st.error("❌ SwissDock rejected the job. See the response above for the exact reason.")

        # Parse the reason and give specific fix advice
        reason = param_msg.lower()
        if "empty grid" in reason or "no cav" in reason or "no attractive" in reason:
            st.markdown("""
<div class="warn-box">
<b>Reason: No binding cavity found in your grid box.</b><br><br>
<b>Fix options (try in order):</b><br>
1. ✅ Enable <b>Blind docking</b> in the sidebar — no box needed at all<br>
2. Check your X/Y/Z center: open the PDB in a viewer (e.g. UCSF Chimera or Mol* on RCSB),
   hover over the binding site residues, and note the coordinates<br>
3. Set cavity depth to <b>50 (shallow)</b> in the sidebar — it detects more surface clefts<br>
4. Try a smaller box size (e.g. 20×20×20) centered exactly on the active site
</div>""", unsafe_allow_html=True)

        elif "too long" in reason or "time" in reason:
            st.markdown("""
<div class="warn-box">
<b>Reason: Estimated calculation time too long.</b><br><br>
<b>Fix options:</b><br>
1. Switch to <b>AutoDock Vina</b> engine (faster) with box ≤ 30×30×30 Å<br>
2. Reduce exhaustiveness to 1 or 2<br>
3. Reduce box size (each dimension ≤ 25 Å for Vina, ≤ 40 Å for AC)<br>
4. If using AC: reduce sampling exhaustivity to 180° (fastest)
</div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
<div class="warn-box">
<b>Exact SwissDock message above tells you the reason.</b><br>
Common fixes: enable Blind docking, reduce box size, reduce exhaustiveness.
</div>""", unsafe_allow_html=True)
        st.stop()

    log("Step 3/5 — ✅ Parameters accepted.")

    # ── Step 4: start
    log("Step 4/5 — Submitting job to SwissDock queue…")
    ok, start_msg = sd_start(session_id)
    if not ok:
        st.error(f"❌ Could not start docking:\n```\n{start_msg}\n```")
        st.stop()
    log("Step 4/5 — ✅ Job queued. Polling every 15 s…")

    # ── Step 5: poll
    bar  = st.progress(0, "Waiting in queue…")
    slot = st.empty()

    for i in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        status = sd_status(session_id)
        slot.info(f"⏳ {status}")
        bar.progress(min((i+1)/MAX_POLLS, 0.99), f"Poll {i+1}: {status[:70]}")

        if "finished" in status.lower():
            bar.progress(1.0, "✅ Done!")
            log("Step 5/5 — ✅ Finished! Downloading results…")
            tb = sd_retrieve(session_id)
            if tb:
                st.session_state.tarball = tb
                st.session_state.done    = True
            else:
                st.error("Could not download results. Use Fetch Results button.")
            break
        if "error" in status.lower() or "fail" in status.lower():
            st.error(f"SwissDock error: {status}")
            break
    else:
        st.warning("Job is still running after 60 min. Use Fetch Results when done.")

# Manual controls
if poll_btn and st.session_state.sd_session:
    st.info(sd_status(st.session_state.sd_session))

if cancel_btn and st.session_state.sd_session:
    st.warning(sd_cancel(st.session_state.sd_session))
    st.session_state.sd_session = None

if fetch_btn and st.session_state.sd_session:
    with st.spinner("Fetching…"):
        tb = sd_retrieve(st.session_state.sd_session)
    if tb:
        st.session_state.tarball = tb
        st.session_state.done    = True
        st.success("Results fetched!")
    else:
        st.error("Not ready yet — try again shortly.")


# ─────────────────────────────────────────────────────────────────────────────
#  Results display
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.done and st.session_state.tarball:
    st.markdown("---")
    st.markdown('<div class="sec-hdr">📊 Docking Results</div>', unsafe_allow_html=True)
    parsed = parse_tarball(st.session_state.tarball)

    st.download_button(
        "⬇️ Download full results (.tar.gz)",
        data=st.session_state.tarball,
        file_name=f"swissdock_{st.session_state.sd_session}.tar.gz",
        mime="application/gzip",
    )
    with st.expander("📁 Files in archive"):
        st.write(parsed["files"])

    if parsed["scores"] is not None:
        st.subheader("📈 Docking Scores")
        st.dataframe(parsed["scores"], use_container_width=True)
    elif parsed["log"]:
        st.subheader("📋 Raw Log")
        st.code(parsed["log"])

    if parsed["poses"]:
        fmt = "pdbqt" if "REMARK VINA" in parsed["poses"] else "pdb"
        with st.expander("🔬 Docked Poses — 3D Viewer", expanded=True):
            components.html(render_3d(parsed["poses"], fmt, 450), height=470)


# ─────────────────────────────────────────────────────────────────────────────
#  Welcome / help
# ─────────────────────────────────────────────────────────────────────────────
if not receptor_file and not ligand_file and not smiles_input:
    st.info(
        "**👈 Upload structures in the sidebar to begin.**\n\n"
        "**Quick start (easiest):**\n"
        "1. Upload Receptor PDB + Ligand SDF\n"
        "2. Tick **Blind docking** (ON by default) — no box needed\n"
        "3. Click **🚀 Submit to SwissDock**\n\n"
        "**If you know the binding site:**\n"
        "Untick Blind docking → enter Box Center X/Y/Z from literature/PDB viewer\n\n"
        "> SwissDock is free for academic use. Docking runs on SIB servers in Switzerland."
    )

st.markdown("---")
st.caption("MasterDock · SwissDock REST API · 3Dmol.js · RDKit · Biopython")
