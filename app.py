# -*- coding: utf-8 -*-
"""
MasterDock – Molecular Interaction Analysis Platform
Docking engine : DiffDock-L via Neurosnap REST API (neurosnap.ai)
                 Works from any cloud IP, no firewall issues.
3D viewer      : 3Dmol.js via CDN (no IPython needed)
2D viewer      : RDKit
Parsing        : Biopython (PDB) + RDKit (SDF)
"""

import io
import json
import time

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
from requests_toolbelt.multipart.encoder import MultipartEncoder

# ─────────────────────────────────────────────────────────────────────────────
#  Page config  — MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MasterDock · DiffDock-L",
    page_icon="🧬",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Neurosnap API constants
# ─────────────────────────────────────────────────────────────────────────────
NS_BASE       = "https://neurosnap.ai/api"
NS_SUBMIT     = f"{NS_BASE}/job/submit/DiffDock-L"
NS_STATUS     = f"{NS_BASE}/job/status"
NS_DATA       = f"{NS_BASE}/job/data"
NS_FILE       = f"{NS_BASE}/job/file"
NS_CANCEL     = f"{NS_BASE}/job/cancel"

POLL_INTERVAL = 20   # seconds between polls
MAX_POLLS     = 180  # 180 × 20 s = 60 min max

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
.api-log{background:#f0f4ff;border-left:4px solid #3949ab;padding:8px 14px;
  border-radius:4px;font-size:.83rem;margin-bottom:6px}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  3-D viewer — 3Dmol.js CDN (no IPython / no py3Dmol needed)
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


# ─────────────────────────────────────────────────────────────────────────────
#  Neurosnap DiffDock-L API client
# ─────────────────────────────────────────────────────────────────────────────

def ns_headers(api_key: str) -> dict:
    return {"X-API-KEY": api_key}


def ns_submit_diffdock(api_key: str, pdb_content: str, smiles: str,
                        num_poses: int = 10):
    """
    Submit DiffDock-L to Neurosnap.
    CONFIRMED WORKING: receptor = file tuple, ligand = JSON list
    UNKNOWN: exact field name + value for Number Samples
    Tries all combinations.
    Returns (success, job_id_or_err, all_responses).
    """
    url = NS_SUBMIT + "?note=MasterDock"
    hdr = ns_headers(api_key)
    pdb_bytes = pdb_content.encode("utf-8")

    try:
        from rdkit import Chem as _C
        _m = _C.MolFromSmiles(smiles)
        clean_smiles = _C.MolToSmiles(_m) if _m else smiles
    except Exception:
        clean_smiles = smiles

    receptor_field = ("receptor.pdb", pdb_bytes, "chemical/x-pdb")
    ligand_field   = json.dumps([{"data": clean_smiles, "type": "smiles"}])

    # Try combinations of (field_name, field_value)
    combos = [
        ("Number Samples", "10"),
        ("Number Samples", "20"),
        ("Number Samples", "40"),
        ("Number Samples", "10 Samples"),
        ("Number Samples", "20 Samples"),
        ("Number Samples", "40 Samples"),
        ("Number Samples", "Low (10)"),
        ("Number Samples", "Medium (20)"),
        ("Number Samples", "High (40)"),
        ("Num Samples",    "10"),
        ("Num Samples",    "20"),
        ("Num Samples",    "40"),
        ("num_samples",    "10"),
        ("num_poses",      "10"),
        ("Number of Samples", "10"),
        ("Number of Poses",   "10"),
    ]

    responses = []
    for field_name, field_val in combos:
        fields = {
            "Input Receptor": receptor_field,
            "Input Ligand":   ligand_field,
            field_name:       field_val,
        }
        try:
            mp = MultipartEncoder(fields=fields)
            r  = requests.post(url,
                               headers={**hdr, "Content-Type": mp.content_type},
                               data=mp, timeout=60)
            full = r.text.strip()
            label = f"({repr(field_name)}, {repr(field_val)}) → HTTP {r.status_code}: {full}"
            responses.append(label)
            if r.status_code == 200:
                return True, str(r.json()).strip('"\' '), responses
            if r.status_code in (401, 403):
                return False, f"Auth/IP error: {full}", responses
        except Exception as e:
            responses.append(f"({repr(field_name)}, {repr(field_val)}) → Exception: {e}")

    return False, "All combinations failed — see responses for details", responses

def ns_job_status(api_key: str, job_id: str) -> str:
    """Returns status string: pending | running | completed | failed | cancelled"""
    try:
        r = requests.get(f"{NS_STATUS}/{job_id}",
                         headers=ns_headers(api_key), timeout=30)
        return r.json() if r.status_code == 200 else f"error:{r.status_code}"
    except Exception as e:
        return f"error:{e}"


def ns_job_files(api_key: str, job_id: str) -> dict:
    """Get list of output files for a completed job."""
    try:
        r = requests.get(f"{NS_DATA}/{job_id}",
                         headers=ns_headers(api_key), timeout=30)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def ns_download_file(api_key: str, job_id: str, filename: str) -> bytes | None:
    """Download a specific output file from a completed job."""
    try:
        r = requests.get(f"{NS_FILE}/{job_id}/out/{filename}",
                         headers=ns_headers(api_key), timeout=120)
        return r.content if r.status_code == 200 else None
    except Exception:
        return None


def ns_cancel_job(api_key: str, job_id: str) -> str:
    try:
        r = requests.post(f"{NS_CANCEL}/{job_id}",
                          headers=ns_headers(api_key), timeout=30)
        return "Cancelled" if r.status_code == 200 else r.text
    except Exception as e:
        return str(e)


def parse_diffdock_results(job_data: dict, api_key: str, job_id: str):
    """
    Download and parse DiffDock-L output files.
    Returns dict with keys: poses (list of SDF strings), scores (DataFrame), raw_files
    """
    out_files = [f[0] for f in job_data.get("out", [])]
    result = {"poses": [], "scores": None, "raw_files": out_files}

    scores = []
    for fname in out_files:
        data = ns_download_file(api_key, job_id, fname)
        if data is None:
            continue

        fname_lower = fname.lower()

        # Confidence / score JSON
        if "confidence" in fname_lower and fname_lower.endswith(".json"):
            try:
                conf = json.loads(data.decode("utf-8"))
                if isinstance(conf, list):
                    for i, c in enumerate(conf):
                        scores.append({"Pose": i + 1, "Confidence": round(float(c), 4)})
                elif isinstance(conf, dict):
                    for k, v in conf.items():
                        scores.append({"Pose": k, "Confidence": round(float(v), 4)})
            except Exception:
                pass

        # SDF pose files (rank_1.sdf, rank_2.sdf …)
        elif fname_lower.endswith(".sdf"):
            try:
                result["poses"].append(data.decode("utf-8", errors="ignore"))
            except Exception:
                pass

    if scores:
        result["scores"] = pd.DataFrame(scores).sort_values("Confidence", ascending=False)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────
for k, v in {
    "ns_job_id": None,
    "docking_done": False,
    "result": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 MasterDock")
    st.caption("DiffDock-L · Neurosnap API · 3Dmol.js · RDKit")
    st.markdown("---")

    # API key
    st.markdown("### 🔑 Neurosnap API Key")
    api_key = st.text_input(
        "Enter your Neurosnap API key",
        type="password",
        placeholder="Paste your key here",
        help="Get a free key at neurosnap.ai → Overview → API tab",
    )
    if not api_key:
        st.markdown(
            '<div style="background:#fff3e0;border-left:3px solid #e65100;'
            'padding:8px;border-radius:4px;font-size:.82rem;color:#bf360c">'
            '1. Register free at <b>neurosnap.ai</b><br>'
            '2. Go to Overview → API tab<br>'
            '3. Click Generate API Key<br>'
            '4. Paste it above</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📂 Upload Structures")
    receptor_file = st.file_uploader("Receptor PDB", type=["pdb"], key="rec")

    lig_mode = st.radio("Ligand input", ["Enter SMILES", "Upload SDF"], horizontal=True)
    if lig_mode == "Enter SMILES":
        smiles_input = st.text_input(
            "Ligand SMILES",
            placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
            help="DiffDock-L accepts SMILES directly — no file conversion needed",
        )
        ligand_file = None
    else:
        ligand_file = st.file_uploader("Ligand SDF", type=["sdf"], key="lig")
        smiles_input = None

    st.markdown("---")
    st.markdown("### ⚙️ DiffDock-L Settings")
    num_poses = st.selectbox(
        "Number of poses",
        [10, 20, 40],
        index=0,
        help="More poses = better coverage but longer runtime",
    )

    st.markdown("---")
    can_run = bool(api_key and receptor_file and (smiles_input or ligand_file))
    submit_btn = st.button(
        "🚀 Run DiffDock-L",
        disabled=not can_run,
        use_container_width=True,
        type="primary",
    )

    if st.session_state.ns_job_id:
        st.markdown("---")
        st.info(f"Job: `{st.session_state.ns_job_id}`")
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
    "Upload your **Receptor PDB** and **Ligand SMILES** in the sidebar, "
    "then click **🚀 Run DiffDock-L**."
)

receptor_content = None
ligand_mols      = []
final_smiles     = None

# ── Receptor ─────────────────────────────────────────────────────────────────
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

# ── Ligand ────────────────────────────────────────────────────────────────────
if lig_mode == "Enter SMILES" and smiles_input and smiles_input.strip():
    st.markdown("---")
    st.markdown('<div class="sec-hdr">🟢 Ligand: SMILES</div>', unsafe_allow_html=True)
    mol = Chem.MolFromSmiles(smiles_input.strip())
    if mol:
        final_smiles = smiles_input.strip()
        st.success("✅ Valid SMILES")
        col1, col2 = st.columns(2)
        col1.json(mol_props(mol))
        img = render_2d(mol)
        if img:
            col2.image(img, caption="Ligand 2D structure", width=240)
    else:
        st.error("❌ Invalid SMILES — check the string and try again.")

elif ligand_file is not None:
    st.markdown("---")
    st.markdown(f'<div class="sec-hdr">🟢 Ligand: {ligand_file.name}</div>',
                unsafe_allow_html=True)
    lig_txt = ligand_file.read().decode("utf-8")
    ligand_mols = parse_sdf(lig_txt)
    if ligand_mols:
        mol = ligand_mols[0]
        final_smiles = Chem.MolToSmiles(mol)
        st.write(f"Found **{len(ligand_mols)}** molecule(s). Using first molecule.")
        st.dataframe(
            pd.DataFrame([mol_props(m) for m in ligand_mols],
                         index=[f"Mol {i+1}" for i in range(len(ligand_mols))]),
            use_container_width=True,
        )
        with st.expander("🔬 Ligand 3D Viewer", expanded=True):
            components.html(render_3d(lig_txt, "sdf", 340), height=360)
        n = min(len(ligand_mols), 4)
        cols = st.columns(n)
        for col, m in zip(cols, ligand_mols[:n]):
            img = render_2d(m)
            if img:
                col.image(img, use_container_width=True)
        if final_smiles:
            st.caption(f"SMILES sent to DiffDock-L: `{final_smiles}`")
    else:
        st.error("No valid molecules found in SDF file.")


# ─────────────────────────────────────────────────────────────────────────────
#  DiffDock-L submission
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="sec-hdr">⚗️ DiffDock-L Docking (Neurosnap)</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="tip-box">
<b>DiffDock-L</b> is a state-of-the-art deep learning docking model from MIT (Barzilay Lab).
It needs no grid box, no binding site coordinates, and no PDBQT conversion —
just a PDB file and a SMILES string. It runs on Neurosnap's GPU servers and works
reliably from any cloud IP.
</div>""", unsafe_allow_html=True)

if submit_btn:
    if not api_key:
        st.error("Please enter your Neurosnap API key in the sidebar.")
        st.stop()
    if not receptor_content:
        st.error("Please upload a receptor PDB file.")
        st.stop()
    if not final_smiles:
        st.error("Please provide a ligand (SMILES string or SDF file).")
        st.stop()

    st.session_state.docking_done = False
    st.session_state.result       = None
    log_slot = st.empty()

    def log(msg):
        log_slot.markdown(f'<div class="api-log">🔹 {msg}</div>',
                          unsafe_allow_html=True)

    # Submit
    log("Submitting DiffDock-L job to Neurosnap…")
    ok, result, all_responses = ns_submit_diffdock(api_key, receptor_content, final_smiles, num_poses)

    if not ok:
        st.error("❌ Submission failed — all formats tried")
        r_low = result.lower()
        if "401" in result or "403" in result or "auth" in r_low or "invalid" in r_low:
            st.warning("🔑 **API key / IP issue** — check your key at neurosnap.ai → Overview → API tab.")
        elif "credits" in r_low or "payment" in r_low:
            st.warning("💳 **Insufficient credits** — top up your Neurosnap account at neurosnap.ai.")
        with st.expander("📋 Full server responses (click to debug)", expanded=True):
            for resp in all_responses:
                st.code(resp, language="text")
        st.info(
            "📧 **Next step:** Copy the server responses above and email them to "
            "**hello@neurosnap.ai** — ask for the correct multipart field format "
            "for the DiffDock-L API endpoint `/api/job/submit/DiffDock-L`."
        )
        st.stop()

    job_id = result
    st.session_state.ns_job_id = job_id
    log(f"✅ Job submitted! ID: **`{job_id}`**")
    st.info(f"Job ID: `{job_id}` — you can also track this at neurosnap.ai → Overview → Jobs")

    # Poll
    bar  = st.progress(0, "Job submitted — waiting to start…")
    slot = st.empty()

    for i in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        status = ns_job_status(api_key, job_id)
        slot.info(f"⏳ Status: **{status}**  (poll {i+1}/{MAX_POLLS})")
        bar.progress(min((i + 1) / MAX_POLLS, 0.99),
                     f"Status: {status} — poll {i+1}")

        if status == "completed":
            bar.progress(1.0, "✅ Completed!")
            log("✅ Job complete! Downloading results…")

            job_data = ns_job_files(api_key, job_id)
            parsed   = parse_diffdock_results(job_data, api_key, job_id)
            st.session_state.result       = parsed
            st.session_state.docking_done = True
            break

        elif status in ("failed", "cancelled", "deleted"):
            st.error(f"❌ Job ended with status: **{status}**")
            st.info("Check neurosnap.ai → Overview → Jobs for details.")
            st.stop()

        elif status.startswith("error:"):
            st.warning(f"Status check error: {status}. Will retry…")
    else:
        st.warning(
            "Job is taking longer than 60 minutes. "
            "Use the **Fetch Results** button once it finishes on neurosnap.ai."
        )

# Manual buttons
if poll_btn and st.session_state.ns_job_id and api_key:
    s = ns_job_status(api_key, st.session_state.ns_job_id)
    st.info(f"Current status: **{s}**")

if cancel_btn and st.session_state.ns_job_id and api_key:
    msg = ns_cancel_job(api_key, st.session_state.ns_job_id)
    st.warning(f"Cancel response: {msg}")

if fetch_btn and st.session_state.ns_job_id and api_key:
    with st.spinner("Fetching results from Neurosnap…"):
        job_data = ns_job_files(api_key, st.session_state.ns_job_id)
        parsed   = parse_diffdock_results(job_data, api_key, st.session_state.ns_job_id)
    if parsed["poses"] or parsed["scores"] is not None:
        st.session_state.result       = parsed
        st.session_state.docking_done = True
        st.success("Results fetched!")
    else:
        st.error("No results yet — job may still be running.")


# ─────────────────────────────────────────────────────────────────────────────
#  Results display
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.docking_done and st.session_state.result:
    res = st.session_state.result
    st.markdown("---")
    st.markdown('<div class="sec-hdr">📊 DiffDock-L Results</div>',
                unsafe_allow_html=True)

    # Files list
    with st.expander(f"📁 Output files ({len(res['raw_files'])})"):
        st.write(res["raw_files"])

    # Confidence scores
    if res["scores"] is not None:
        st.subheader("📈 Pose Confidence Scores")
        st.caption("Higher confidence = better predicted binding pose")
        st.dataframe(res["scores"], use_container_width=True)

        # Highlight best
        best = res["scores"].iloc[0]
        conf = best.get("Confidence", "N/A")
        if isinstance(conf, float):
            if conf > 0:
                st.success(f"🎯 Best pose confidence: **{conf:.4f}** (positive = good prediction)")
            else:
                st.warning(f"⚠️ Best pose confidence: **{conf:.4f}** (negative = uncertain prediction)")

    # 3-D viewer — show top pose
    if res["poses"]:
        st.subheader("🔬 Docked Poses — 3D Viewer")
        pose_idx = st.selectbox(
            "Select pose to view",
            list(range(1, len(res["poses"]) + 1)),
            format_func=lambda x: f"Pose {x}",
        ) - 1

        selected_pose = res["poses"][pose_idx]
        with st.expander(f"Pose {pose_idx + 1} — 3D Viewer", expanded=True):
            # Show receptor + docked pose together if possible
            if receptor_content:
                combined = receptor_content + "\n" + selected_pose
                components.html(render_3d(combined, "pdb", 480), height=500)
            else:
                components.html(render_3d(selected_pose, "sdf", 480), height=500)

        # 2-D of top pose
        pose_mols = parse_sdf(selected_pose)
        if pose_mols:
            img = render_2d(pose_mols[0])
            if img:
                st.image(img, caption=f"Pose {pose_idx + 1} — 2D structure", width=280)

        # Download all poses
        all_poses_sdf = "\n$$$$\n".join(res["poses"])
        st.download_button(
            "⬇️ Download all poses (.sdf)",
            data=all_poses_sdf.encode(),
            file_name=f"diffdock_poses_{st.session_state.ns_job_id}.sdf",
            mime="chemical/x-mdl-sdfile",
        )

    elif not res["scores"] is not None:
        st.warning("Results were retrieved but no poses or scores were found. "
                   "Check your job on neurosnap.ai for details.")


# ─────────────────────────────────────────────────────────────────────────────
#  Welcome / help
# ─────────────────────────────────────────────────────────────────────────────
if not receptor_file and not smiles_input and not ligand_file:
    st.info(
        "**👈 Upload your structures in the sidebar to begin.**\n\n"
        "**How to use MasterDock:**\n"
        "1. Get a **free Neurosnap API key** at neurosnap.ai → Overview → API\n"
        "2. Upload your **Receptor PDB** file\n"
        "3. Enter your **Ligand SMILES** (or upload an SDF)\n"
        "4. Click **🚀 Run DiffDock-L**\n"
        "5. Results appear automatically — confidence scores + 3D poses\n\n"
        "**Why DiffDock-L?**\n"
        "- No grid box or binding site coordinates needed\n"
        "- No PDBQT conversion, no OpenBabel required\n"
        "- Works from any cloud server (no IP blocking)\n"
        "- State-of-the-art accuracy (MIT Barzilay Lab)\n\n"
        "> Neurosnap gives free credits on registration. "
        "Each DiffDock-L job typically costs a few credits."
    )

st.markdown("---")
st.caption(
    "MasterDock · Docking: DiffDock-L via Neurosnap API · "
    "3D: 3Dmol.js · Cheminformatics: RDKit · Parsing: Biopython · "
    "[DiffDock-L paper](https://arxiv.org/abs/2402.18396)"
)
