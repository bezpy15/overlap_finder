import os, re, json, io
import streamlit as st
import pandas as pd, numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:
    matplotlib = None
    plt = None
    LinearSegmentedColormap = None
    MATPLOTLIB_AVAILABLE = False
from utils.enrichment import run_enrichr_generic, build_overlap_table
from utils.opentargets import fetch_targets  # your retried+timeout version

# -------------------------------- Config --------------------------------
st.set_page_config(page_title="BHB–Disease Reactome Overlap", layout="wide")

# Tidy metrics & align numerals
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
[data-testid="stMetric"] { min-height: 110px; }
</style>
""", unsafe_allow_html=True)

TITLE = "BHB–Disease Reactome Overlap"
LIBRARIES = ["Reactome_2022"]
TERMS_METADATA_PATH = "data/reactome_pathways_hierarchy_genes.csv"
BHB_SEEDS_PATH = "data/bhb_seeds_with_tissue_with_refs.csv"
HUBNESS_PATH_CANDIDATES = [
    "data/hub_calls_per_gene.csv",                # preferred (with reactome_id)
    "data/reactome_term_hubness_per_gene.csv",    # also supported
]
TISSUE_ENRICHMENT_PATH = "data/tissue_enrichment_results.csv"
GTEX_WIDE_PATH = "data/GTEx_median_TPM_WIDE.csv"

CODE_TISSUES = [
    "Adipose_Subcutaneous",
    "Adipose_Visceral_Omentum",
    "Artery_Aorta",
    "Brain_Cortex",
    "Brain_Hippocampus",
    "Brain_Frontal_Cortex_BA9",
    "Brain_Spinal_cord_cervical_c-1",
    "Cervix_Endocervix",
    "Colon_Sigmoid",
    "Fallopian_Tube",
    "Heart_Left_Ventricle",
    "Kidney_Cortex",
    "Liver",
    "Muscle_Skeletal",
    "Nerve_Tibial",
    "Pancreas",
    "Prostate",
    "Small_Intestine_Terminal_Ileum",
    "Spleen",
    "Stomach",
    "Testis",
    "Thyroid",
    "Uterus",
    "Vagina",
    "Whole_Blood",
    "Lung",
]

_BLUES_HEX = ["#ffffff","#deebf7","#c6dbef","#9ecae1","#6baed6","#4292c6","#2171b5","#084594"]
if MATPLOTLIB_AVAILABLE:
    BLUES_CMAP = LinearSegmentedColormap.from_list("blues_custom", _BLUES_HEX, N=256)
    BLUES_CMAP.set_bad("#f2f2f2")
else:
    BLUES_CMAP = None

# ----------------------------- Helpers -----------------------------------
ID_REGEX = re.compile(r"R-[A-Z]+-\d+")

def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def parse_enrichr_term(term: str):
    """Return (stable_id, clean_name) from Enrichr term string."""
    if term is None or (isinstance(term, float) and np.isnan(term)):
        return None, None
    s = str(term)
    m = ID_REGEX.search(s)
    stable_id = m.group(0) if m else None
    name = s
    if stable_id:
        name = re.sub(re.escape(stable_id), "", name)
        name = re.sub(r"[\(\)\[\]\{\}:–—\-|]+", " ", name)
    else:
        name = re.sub(r"[\(\)\[\]\{\}]+", " ", name)
    name = re.sub(r"^\s*[:\-–—]+\s*", " ", name)
    name = re.sub(r"\s{2,}", " ", name).strip()
    return stable_id, (name if name else None)

def _format_parent_paths(raw) -> str | None:
    """Format a breadcrumb from various encodings."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            parts = [str(x).strip() for x in obj if str(x).strip()]
            return " › ".join(parts) if parts else None
    except Exception:
        pass
    for sep in [" > ", ">>", "→", "->", "—", " - ", "||", "|", ";", ",", "/", "\\", "›"]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            return " › ".join(parts) if parts else None
    return s

def load_terms_metadata(path: str):
    """Return maps for sizes & hierarchy + slider bounds."""
    try:
        meta = pd.read_csv(path)
    except Exception:
        return {}, {}, {}, {}, None, None

    name_col = next((c for c in meta.columns
                     if c in ["displayName","name","term","description","reactome_term","reactome name","pathway"]), meta.columns[0])
    id_col = next((c for c in meta.columns
                   if c in ["stId","stable_id","reactome_id","reactomeId","id","st_id"]), None)
    n_col = next((c for c in meta.columns
                  if c in ["n_genes","gene_count","genes_in_term","nGenes","n_genes_total","term_size","gene_list_size","n"]), None)
    parent_col = next((c for c in meta.columns
                       if c in ["parent_paths","parent_path","parents","ancestors","hierarchy","path","parentHierarchy"]), None)

    if n_col is None:
        list_col = next((c for c in meta.columns
                         if c in ["gene_list","genes","all_genes_in_term","all_genes","members"]), None)
        if list_col is not None:
            def _count(x):
                if pd.isna(x): return np.nan
                toks = [t for t in re.split(r"[;,\s]+", str(x)) if t]
                return len(toks) if toks else np.nan
            meta["n_genes"] = meta[list_col].apply(_count)
            n_col = "n_genes"
        else:
            meta["n_genes"] = np.nan
            n_col = "n_genes"

    if id_col is not None:
        meta["_id_key"] = meta[id_col].astype(str).str.strip().str.upper()
    meta["_name_key"] = _norm(meta[name_col])

    id2n = meta.dropna(subset=["_id_key"])[["_id_key", n_col]].set_index("_id_key")[n_col].to_dict() if id_col else {}
    name2n = meta[["_name_key", n_col]].set_index("_name_key")[n_col].to_dict()

    id2path = meta.dropna(subset=["_id_key"])[["_id_key", parent_col]].set_index("_id_key")[parent_col].to_dict() if (id_col and parent_col) else {}
    name2path = meta[["_name_key", parent_col]].set_index("_name_key")[parent_col].to_dict() if parent_col else {}

    nn = pd.to_numeric(meta[n_col], errors="coerce").dropna()
    min_n = int(nn.min()) if not nn.empty else None
    max_n = int(nn.max()) if not nn.empty else None
    return id2n, name2n, id2path, name2path, min_n, max_n

def truthy(x) -> bool:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"1","true","t","yes","y","hub"}

def load_hubness_maps_with_id(path_candidates):
    """
    CSV must have:
      - reactome_id (primary key, e.g., R-HSA-1643685)
      - gene
      - is_hub
      - term (optional; fallback)
    """
    df = None
    for p in path_candidates:
        try:
            df_try = pd.read_csv(p)
            df = df_try
            break
        except Exception:
            continue
    if df is None:
        return {}, {}, {}

    id_col = next((c for c in df.columns if c.lower() in ["reactome_id","stid","stable_id","reactomeid","st_id","id"]), None)
    term_col = next((c for c in df.columns if c.lower() in ["term","displayname","name","description","pathway"]), None)
    gene_col = next((c for c in df.columns if c.lower() in ["gene","gene_symbol","symbol","hgnc_symbol","approved_symbol"]), None)
    hub_col  = next((c for c in df.columns if c.lower() in ["is_hub","hub","ishub","is_hub_gene","is_hub_bool"]), None)

    if id_col is None or gene_col is None or hub_col is None:
        return {}, {}, {}

    if id_col is not None:
        df["_id_key"] = df[id_col].astype(str).str.strip().str.upper()
    else:
        df["_id_key"] = np.nan

    if term_col is not None:
        df["_term_raw"] = df[term_col].astype(str).str.strip()
        parsed = df["_term_raw"].apply(parse_enrichr_term)
        df["_term_name"] = parsed.apply(lambda t: t[1])
        df["_name_key"] = _norm(df["_term_name"].fillna(df["_term_raw"]))
    else:
        df["_term_raw"] = np.nan
        df["_name_key"] = np.nan

    df["_gene"] = df[gene_col].astype(str).str.strip().str.upper()
    df["_is_hub"] = df[hub_col].apply(truthy)

    df_h = df[df["_is_hub"]].copy()

    hubs_by_reactome_id = {}
    for k, sub in df_h.dropna(subset=["_id_key"]).groupby("_id_key"):
        hubs_by_reactome_id[k] = set(sub["_gene"].tolist())

    hubs_by_raw_term = {}
    if term_col is not None:
        for k, sub in df_h.dropna(subset=["_term_raw"]).groupby("_term_raw"):
            hubs_by_raw_term[k] = set(sub["_gene"].tolist())

    hubs_by_name = {}
    if term_col is not None:
        for k, sub in df_h.dropna(subset=["_name_key"]).groupby("_name_key"):
            hubs_by_name[k] = set(sub["_gene"].tolist())

    return hubs_by_reactome_id, hubs_by_raw_term, hubs_by_name


def load_tissue_enrichment_map(path: str) -> dict[str, str]:
    """Map Reactome stable IDs to semicolon-joined enriched tissues (fold.change > 0)."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    id_col = next((
        c for c in df.columns
        if re.sub(r"[_.]", "", c).lower() in {"reactomeid", "stid", "stableid"}
    ), None)
    tissue_col = next((c for c in df.columns if c.lower() == "tissue"), None)
    if tissue_col is None:
        tissue_col = next((c for c in df.columns if "tissue" in c.lower()), None)
    fold_col = next((c for c in df.columns if "fold" in c.lower()), None)

    if id_col is None or tissue_col is None or fold_col is None:
        return {}

    df["_id_key"] = df[id_col].astype(str).str.strip().str.upper()
    df["_tissue"] = df[tissue_col].astype(str).str.strip()
    df["_fold"] = pd.to_numeric(df[fold_col], errors="coerce")

    df_valid = df[
        (df["_id_key"].ne(""))
        & (df["_tissue"].ne(""))
        & (df["_fold"].notna())
        & (df["_fold"] > 0)
    ].copy()

    tissues_by_id: dict[str, str] = {}
    for rid, sub in df_valid.groupby("_id_key"):
        seen = set()
        tissues: list[str] = []
        sub_sorted = sub.sort_values("_fold", ascending=False)
        for tissue in sub_sorted["_tissue"]:
            if tissue in seen:
                continue
            seen.add(tissue)
            tissues.append(tissue)
        if tissues:
            tissues_by_id[rid] = ";".join(tissues)

    return tissues_by_id


ENS_RE = re.compile(r"^ENSG[0-9]+$", re.IGNORECASE)


def _clean_ens_ids(seq) -> list[str]:
    vals = []
    for item in seq:
        if item is None:
            continue
        s = str(item).strip().upper()
        if not s:
            continue
        s = re.sub(r'^"|"$', "", s)
        s = re.sub(r"^'|'$", "", s)
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"\..*$", "", s)
        if ENS_RE.match(s):
            vals.append(s)
    out = []
    seen = set()
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def split_ens_gene_list(raw) -> list[str]:
    if not isinstance(raw, str):
        return []
    s = raw.strip()
    if not s:
        return []
    tokens = re.split(r"[,;\|]+", s)
    return _clean_ens_ids(tokens)


def load_reactome_ens_map(path: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}, {}

    id_col = next((c for c in df.columns if c in ["stId","stable_id","reactome_id","reactomeId","id","st_id"]), None)
    name_col = next((c for c in df.columns if c in ["displayName","name","term","description","reactome_term","reactome name","pathway"]), None)
    ens_col = next((c for c in df.columns if c.lower() in {"ens_gene_list","ensg_list","ensgenes","ens_genes"}), None)

    if ens_col is None:
        return {}, {}

    df["_ens_list"] = df[ens_col].apply(split_ens_gene_list)
    if id_col is not None:
        df["_id_key"] = df[id_col].astype(str).str.strip().str.upper()
    else:
        df["_id_key"] = ""
    if name_col is not None:
        df["_name_key"] = _norm(df[name_col])
    else:
        df["_name_key"] = ""

    ens_by_id: dict[str, list[str]] = {}
    for rid, sub in df[df["_id_key"].ne("")].groupby("_id_key"):
        lists = [item for lst in sub["_ens_list"] for item in lst]
        ens_by_id[rid] = _clean_ens_ids(lists)

    ens_by_name: dict[str, list[str]] = {}
    for name_key, sub in df[df["_name_key"].ne("")].groupby("_name_key"):
        lists = [item for lst in sub["_ens_list"] for item in lst]
        ens_by_name[name_key] = _clean_ens_ids(lists)

    return ens_by_id, ens_by_name


@st.cache_data(show_spinner=False)
def load_gtex_wide(path: str) -> pd.DataFrame:
    alt_path = None
    if not os.path.exists(path):
        gz = f"{path}.gz"
        if os.path.exists(gz):
            alt_path = gz
    read_path = alt_path or path
    df = pd.read_csv(read_path)
    if not {"ENSG", "SYMBOL"}.issubset(df.columns):
        raise ValueError("GTEx WIDE CSV must contain columns 'ENSG' and 'SYMBOL'.")
    tissue_cols = [c for c in df.columns if c not in ("ENSG", "SYMBOL")]
    df[tissue_cols] = df[tissue_cols].apply(pd.to_numeric, errors="coerce")
    return df


def _panel_columns(all_columns) -> list[str]:
    cols = [c for c in CODE_TISSUES if c in all_columns]
    if cols:
        return cols
    return [c for c in all_columns if c not in ("ENSG", "SYMBOL")]


def _heatmap_figsize(n_rows: int, n_cols: int) -> tuple[float, float]:
    height = max(4.0, min(18.0, 0.35 * n_rows + 2.5))
    width = max(6.0, min(16.0, 0.5 * n_cols + 3.0))
    return width, height


def build_term_heatmap_png(
    stable_id: str | None,
    name_key: str | None,
    display_term: str,
) -> bytes:
    if not MATPLOTLIB_AVAILABLE or BLUES_CMAP is None:
        raise RuntimeError("Matplotlib is required for heatmap generation (pip install matplotlib).")

    stable_key = (stable_id or "").strip().upper()
    name_key_norm = (name_key or "").strip().lower()

    ens_list = []
    if stable_key and stable_key in ens_lists_by_id:
        ens_list = ens_lists_by_id[stable_key]
    if not ens_list and name_key_norm:
        ens_list = ens_lists_by_name.get(name_key_norm, [])
    if not ens_list and display_term:
        try:
            name_norm = _norm(pd.Series([display_term])).iloc[0]
        except Exception:
            name_norm = display_term.strip().lower()
        ens_list = ens_lists_by_name.get(name_norm, [])
    if not ens_list:
        raise ValueError("No ENS gene list available for this Reactome term.")

    gtex = load_gtex_wide(GTEX_WIDE_PATH)
    panel_cols = _panel_columns(gtex.columns)
    if not panel_cols:
        raise ValueError("No tissue columns available in GTEx expression table.")

    sub = gtex[gtex["ENSG"].isin(ens_list)].copy()
    if sub.empty:
        raise ValueError("No overlap between term ENS genes and GTEx expression library.")

    sub = sub.drop_duplicates(subset=["ENSG"])
    row_labels = sub["SYMBOL"].fillna("")
    row_labels = row_labels.where(row_labels.str.strip().ne(""), sub["ENSG"]).astype(str).to_numpy()

    matrix = np.log2(sub[panel_cols].to_numpy(dtype=float) + 1.0)
    with np.errstate(invalid="ignore"):
        row_means = np.nanmean(matrix, axis=1)
    order = np.argsort(np.nan_to_num(-row_means, nan=0.0))
    matrix = matrix[order, :]
    row_labels = row_labels[order]

    fig_w, fig_h = _heatmap_figsize(matrix.shape[0], matrix.shape[1])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
    data_masked = np.ma.array(matrix, mask=np.isnan(matrix))
    im = ax.imshow(data_masked, aspect="auto", interpolation="nearest", cmap=BLUES_CMAP)

    ax.set_xticks(range(len(panel_cols)))
    ax.set_xticklabels(panel_cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(f"GTEx median TPM: {display_term}", fontsize=12)
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("log2(TPM+1)", rotation=270, va="bottom")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def parse_gene_list(s: str) -> list[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    toks = [t.strip() for t in str(s).split(";")]
    return [t for t in toks if t]

# ----------- BHB evidence (GNN score / evidence / pubs) ------------------
def split_list(val: str) -> list[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    s = str(val).strip()
    if not s:
        return []
    return [t.strip() for t in s.split(";") if t.strip()]

def load_bhb_evidence_map(path: str):
    """
    Returns: dict[symbol_upper] = {
        "gnn_score": float or None,
        "evidence_count": int or None,
        "pmids": [str...],
        "titles": [str...],
        "abstracts": [str...],
    }
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    sym_col = next((c for c in df.columns if c.lower() in ["symbol","gene_symbol","approved_symbol","hgnc_symbol","gene"]), None)
    if sym_col is None:
        sym_col = df.columns[0]

    gnn_col = next((c for c in df.columns if c.lower().replace("_"," ").strip() in ["gnn score","gnnscore"]), None)
    if gnn_col is None and "GNN Score" in df.columns:
        gnn_col = "GNN Score"
    evid_col = next((c for c in df.columns if c.lower().replace("_"," ").strip() in ["evidence count","evidencecount","evidence n","n evidence"]), None)
    if evid_col is None and "evidence_count" in df.columns:
        evid_col = "evidence_count"

    pmid_col = next((c for c in df.columns if "pmid" in c.lower()), None)
    title_col = next((c for c in df.columns if "title" in c.lower()), None)
    abs_col = next((c for c in df.columns if "abstract" in c.lower()), None)

    out = {}
    for _, r in df.iterrows():
        sym = str(r.get(sym_col, "")).strip()
        if not sym:
            continue
        key = sym.upper()
        gnn = r.get(gnn_col, np.nan) if gnn_col else np.nan
        try:
            gnn = float(gnn) if pd.notna(gnn) else None
        except Exception:
            gnn = None
        evid = r.get(evid_col, np.nan) if evid_col else np.nan
        try:
            evid = int(evid) if pd.notna(evid) else None
        except Exception:
            try:
                evid = int(float(evid)) if pd.notna(evid) else None
            except Exception:
                evid = None
        pmids = split_list(r.get(pmid_col, "")) if pmid_col else []
        titles = split_list(r.get(title_col, "")) if title_col else []
        abstracts = split_list(r.get(abs_col, "")) if abs_col else []
        out[key] = {
            "gnn_score": gnn, "evidence_count": evid,
            "pmids": pmids, "titles": titles, "abstracts": abstracts
        }
    return out

def format_score(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)

def format_int(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    try:
        return f"{int(x)}"
    except Exception:
        try:
            return f"{int(float(x))}"
        except Exception:
            return str(x)

def annotate_hub_term_list(hub_set: set[str], disease_set: set[str], bhb_set: set[str]) -> str:
    """List all hubs; * if in one overlap, ** if in both overlaps."""
    if not hub_set:
        return ""
    out = []
    for g in sorted(hub_set):
        in_d = g in disease_set
        in_b = g in bhb_set
        if in_d and in_b:
            out.append(f"{g}**")
        elif in_d or in_b:
            out.append(f"{g}*")
        else:
            out.append(g)
    return ";".join(out)

# ------------------------------ Title -----------------------------------
st.title(TITLE)
st.caption("Pick two gene sets, run Reactome enrichment (via Enrichr), and browse overlapping terms.")

# --------------------------- Sidebar params ------------------------------
P_THRESH = st.sidebar.selectbox(
    "Overlap significance (raw p-value threshold)",
    options=[0.001, 0.01, 0.05],
    index=0,
    help="A Reactome term is considered overlapping only if both gene sets have raw p < threshold for that term."
)
LIBRARY = st.sidebar.selectbox("Enrichment library", options=LIBRARIES, index=0)

# ------------------------ Session state init -----------------------------
st.session_state.setdefault("disease_symbols", [])
st.session_state.setdefault("bhb_symbols", [])
st.session_state.setdefault("disease_source", None)
st.session_state.setdefault("bhb_source", None)
st.session_state.setdefault("results_ready", False)
st.session_state.setdefault("overlaps_raw", None)

# Persist per-term selected gene + expander open state
st.session_state.setdefault("per_term_selected", {})  # {term_key: gene_symbol}
st.session_state.setdefault("term_open", {})          # {term_key: bool}
st.session_state.setdefault("term_heatmap_images", {})  # {term_key: png_bytes}
st.session_state.setdefault("term_force_open", None)
st.session_state.setdefault("scroll_target", None)
st.session_state.setdefault("tissue_filter", [])
st.session_state.setdefault("disease_terms_all", pd.DataFrame())
st.session_state.setdefault("show_all_disease_terms", False)

# --------------------------- Disease selection ---------------------------
st.subheader("1) Disease-associated genes")
src_disease = st.radio(
    "Choose source for disease gene list",
    ["Use predefined CSV (example)", "Fetch via OpenTargets API", "Upload your own CSV"],
    horizontal=True
)

with st.expander("Disease input options", expanded=True):
    if src_disease == "Use predefined CSV (example)":
        path = "data/parkinson_disease_targets_with_scores.csv"
        st.write(f"Predefined file: `{path}`")
        try:
            df = pd.read_csv(path)
            sym_col = next((c for c in df.columns if c.lower() in ['symbol','gene_symbol','approved_symbol','hgnc_symbol']), df.columns[0])
            disease_symbols = (df[sym_col].dropna().astype(str).str.upper().unique().tolist())
            st.session_state.disease_symbols = disease_symbols
            st.session_state.disease_source = "predefined_csv"
            st.write(f"Loaded {len(disease_symbols)} symbols (persisted).")
        except Exception as e:
            st.error(f"Failed to load predefined: {e}")
    elif src_disease == "Fetch via OpenTargets API":
        did = st.text_input("Disease EFO/MONDO ID (e.g., MONDO_0005180 for PD):", "MONDO_0005180")
        any_score = st.slider("Keep genes with any OT score ≥", 0.0, 1.0, 0.5, 0.05)
        col_left, col_right = st.columns([1,3])
        if col_left.button("Fetch from OpenTargets"):
            try:
                df = fetch_targets(did, any_ot_score=any_score)
                sym_col = next((c for c in df.columns if c.lower() in ["symbol","gene_symbol","approved_symbol","hgnc_symbol","targetsymbol"]), None)
                if sym_col is None:
                    raise ValueError(f"Could not find symbol column in OpenTargets result. Columns: {list(df.columns)}")
                disease_symbols = (df[sym_col].dropna().astype(str).str.upper().unique().tolist())
                st.session_state.disease_symbols = disease_symbols
                st.session_state.disease_source = f"opentargets:{did}"
                st.success(f"Fetched & persisted {len(disease_symbols)} symbols from OpenTargets.")
                col_right.dataframe(df.head(30), use_container_width=True)
            except Exception as e:
                st.error(f"OpenTargets error: {e}")
        st.caption(f"Active disease set: {len(st.session_state.disease_symbols)} symbols (source: {st.session_state.disease_source})")
        if st.button("Clear disease set"):
            st.session_state.disease_symbols = []
            st.session_state.disease_source = None
            st.info("Cleared active disease gene set.")
    else:
        up = st.file_uploader("Upload CSV with column 'symbol' (and optional 'score')", type=["csv"])
        if up is not None:
            try:
                df = pd.read_csv(up)
                sym_col = next((c for c in df.columns if c.lower() in ['symbol','gene_symbol','approved_symbol','hgnc_symbol']), df.columns[0])
                disease_symbols = (df[sym_col].dropna().astype(str).str.upper().unique().tolist())
                st.session_state.disease_symbols = disease_symbols
                st.session_state.disease_source = f"upload:{up.name}"
                st.write(f"Loaded {len(disease_symbols)} symbols from upload (persisted).")
            except Exception as e:
                st.error(f"Upload error: {e}")

# ------------------------------ BHB selection ----------------------------
st.subheader("2) BHB-responsive genes")
src_bhb = st.radio(
    "Choose source for BHB gene list",
    ["Use predefined BHB-RG", "Upload your own CSV"],
    horizontal=True
)

with st.expander("BHB input options", expanded=True):
    if src_bhb == "Use predefined BHB-RG":
        path = BHB_SEEDS_PATH
        st.write(f"Predefined file: `{path}`")
        try:
            df = pd.read_csv(path)
            candidate_cols = [c for c in df.columns if c.lower() in ["symbol","gene_symbol","approved_symbol","hgnc_symbol","gene","genesymbol"]]
            col = candidate_cols[0] if candidate_cols else df.columns[0]
            s = df[col].dropna().astype(str)
            s = s[s.str.strip().ne("")].str.split(r"[;\t, ]+").explode().str.strip()
            s = s[s.ne("")].str.replace(r"\s+","", regex=True).str.upper()
            bhb_symbols = s.unique().tolist()
            st.session_state.bhb_symbols = bhb_symbols
            st.session_state.bhb_source = "bhb_predefined_csv"
            st.write(f"Loaded {len(bhb_symbols)} symbols (persisted).")
        except Exception as e:
            st.error(f"Failed to load predefined BHB-RG: {e}")
    else:
        up = st.file_uploader("Upload CSV with column 'gene_symbol' (or 'symbol')", type=["csv"], key="bhb")
        if up is not None:
            try:
                df = pd.read_csv(up)
                sym_col = None
                for c in df.columns:
                    if c.lower() in ["gene_symbol","symbol","hgnc_symbol","approved_symbol","gene"]:
                        sym_col = c; break
                if sym_col is None:
                    sym_col = df.columns[0]
                s = df[sym_col].dropna().astype(str)
                s = s[s.str.strip().ne("")].str.split(r"[;\t, ]+").explode().str.strip()
                s = s[s.ne("")].str.replace(r"\s+","", regex=True).str.upper()
                bhb_symbols = s.unique().tolist()
                st.session_state.bhb_symbols = bhb_symbols
                st.session_state.bhb_source = f"upload:{up.name}"
                st.write(f"Loaded {len(bhb_symbols)} symbols from upload (persisted).")
            except Exception as e:
                st.error(f"BHB upload error: {e}")

# ----------------------- Load term metadata, hubs, & evidence ------------
id2n, name2n, id2path, name2path, meta_min, meta_max = load_terms_metadata(TERMS_METADATA_PATH)
hubs_by_reactome_id, hubs_by_raw_term, hubs_by_name = load_hubness_maps_with_id(HUBNESS_PATH_CANDIDATES)
ens_lists_by_id, ens_lists_by_name = load_reactome_ens_map(TERMS_METADATA_PATH)
enriched_tissues_by_id = load_tissue_enrichment_map(TISSUE_ENRICHMENT_PATH)
bhb_evidence = load_bhb_evidence_map(BHB_SEEDS_PATH)

def _hub_set_for_row(raw_term: str, stable_id: str, name_key: str) -> set[str]:
    """Primary: by reactome_id; fallbacks: raw term, then normalized name."""
    if stable_id and stable_id.upper() in hubs_by_reactome_id:
        return hubs_by_reactome_id[stable_id.upper()]
    if raw_term in hubs_by_raw_term:
        return hubs_by_raw_term[raw_term]
    if name_key in hubs_by_name:
        return hubs_by_name[name_key]
    return set()


def annotate_terms_df(df: pd.DataFrame | None, *, is_overlap: bool) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "p_value_bhb" not in df.columns:
        df["p_value_bhb"] = np.nan
    if "adj_p_value_bhb" not in df.columns:
        df["adj_p_value_bhb"] = np.nan
    if "overlap_count_bhb" not in df.columns:
        df["overlap_count_bhb"] = 0
    if "overlap_genes_bhb" not in df.columns:
        df["overlap_genes_bhb"] = ""
    if "overlap_genes_bhb_annot" not in df.columns:
        df["overlap_genes_bhb_annot"] = df["overlap_genes_bhb"]
    if "overlap_genes_disease_annot" not in df.columns and "overlap_genes_disease" in df.columns:
        df["overlap_genes_disease_annot"] = df["overlap_genes_disease"]

    parsed = df["term"].apply(parse_enrichr_term)
    df["term_stable_id"] = parsed.apply(lambda t: t[0])
    df["term_name_clean"] = parsed.apply(lambda t: t[1])
    df["_term_raw"] = df["term"].astype(str).str.strip()
    df["_id_key"] = df["term_stable_id"].astype(str).str.strip().str.upper()
    df["_name_key"] = _norm(df["term_name_clean"].fillna(df["term"]))

    df["n_genes"] = df["_id_key"].map(id2n)
    mask_missing = df["n_genes"].isna()
    if mask_missing.any():
        df.loc[mask_missing, "n_genes"] = df.loc[mask_missing, "_name_key"].map(name2n)

    df["parent_paths"] = df["_id_key"].map(id2path)
    mask_parent = df["parent_paths"].isna()
    if mask_parent.any():
        df.loc[mask_parent, "parent_paths"] = df.loc[mask_parent, "_name_key"].map(name2path)

    df["enriched_tissues"] = df["_id_key"].map(enriched_tissues_by_id).fillna("")

    df["avg_p_value"] = df[["p_value_disease", "p_value_bhb"]].mean(axis=1, skipna=True)

    disease_lists = df["overlap_genes_disease"].apply(parse_gene_list)
    bhb_lists = df["overlap_genes_bhb_annot"].fillna(df["overlap_genes_bhb"])

    hub_sets, disease_hub_overlap, bhb_hub_overlap, overlapping_hubs, hub_term_annots = [], [], [], [], []
    for raw_term, sid, namek, dis_genes, bhb_annot in zip(
        df["_term_raw"],
        df["term_stable_id"],
        df["_name_key"],
        disease_lists,
        bhb_lists
    ):
        hs = _hub_set_for_row(raw_term, sid, namek)
        hub_sets.append(hs)
        dis_set = {g.strip().upper() for g in dis_genes}
        bhb_set = {g.replace("*", "").strip().upper() for g in str(bhb_annot or "").split(";") if g.strip()}
        disease_hub_overlap.append(len(dis_set & hs))
        bhb_hub_overlap.append(len(bhb_set & hs))
        overlapping_hubs.append(len(dis_set & bhb_set & hs))
        hub_term_annots.append(annotate_hub_term_list(hs, dis_set, bhb_set))

    df["_hub_set"] = hub_sets
    df["disease_hub_overlap"] = disease_hub_overlap
    df["bhb_hub_overlap"] = bhb_hub_overlap
    df["overlapping_hubs"] = overlapping_hubs
    df["hub_genes_term_annot"] = hub_term_annots
    df["is_overlap"] = bool(is_overlap)

    return df.reset_index(drop=True)


def prepare_disease_only_terms(
    df_disease: pd.DataFrame | None,
    df_bhb_all: pd.DataFrame | None,
    p_thresh: float
) -> pd.DataFrame:
    if df_disease is None or df_disease.empty:
        return pd.DataFrame()

    d = df_disease.loc[df_disease["p_value"] < float(p_thresh), [
        "term",
        "p_value",
        "adj_p_value",
        "overlap_count",
        "overlap_genes",
    ]].copy()

    if d.empty:
        return d

    d = d.rename(columns={
        "p_value": "p_value_disease",
        "adj_p_value": "adj_p_value_disease",
        "overlap_count": "overlap_count_disease",
        "overlap_genes": "overlap_genes_disease",
    })

    d["_name_key"] = _norm(d["term"])

    if df_bhb_all is not None and not df_bhb_all.empty:
        bhb = df_bhb_all[[
            "term",
            "p_value",
            "adj_p_value",
            "overlap_count",
            "overlap_genes",
        ]].copy()
        bhb["_name_key"] = _norm(bhb["term"])
        bhb = bhb.sort_values(["p_value", "adj_p_value"], ascending=[True, True], na_position="last")
        bhb = bhb.drop_duplicates(subset="_name_key", keep="first")
        bhb = bhb.rename(columns={
            "p_value": "p_value_bhb",
            "adj_p_value": "adj_p_value_bhb",
            "overlap_count": "overlap_count_bhb",
            "overlap_genes": "overlap_genes_bhb",
        })
        d = d.merge(bhb.drop(columns=["term"]), on="_name_key", how="left")
    else:
        d["p_value_bhb"] = np.nan
        d["adj_p_value_bhb"] = np.nan
        d["overlap_count_bhb"] = 0
        d["overlap_genes_bhb"] = ""

    d["overlap_genes_bhb"] = d.get("overlap_genes_bhb", "").fillna("")
    d["overlap_genes_bhb_annot"] = d["overlap_genes_bhb"]
    d["overlap_genes_disease_annot"] = d["overlap_genes_disease"]
    d["overlap_count_bhb"] = pd.to_numeric(d.get("overlap_count_bhb", 0), errors="coerce").fillna(0).astype(int)
    d["p_value_bhb"] = pd.to_numeric(d.get("p_value_bhb", np.nan), errors="coerce")
    d["adj_p_value_bhb"] = pd.to_numeric(d.get("adj_p_value_bhb", np.nan), errors="coerce")

    return annotate_terms_df(d, is_overlap=False)

# --------------------- Compute overlaps & counters -----------------------
def _compute_and_store_results(d_syms, b_syms, library, p_thresh):
    df_d = run_enrichr_generic(d_syms, library, tag="disease")
    df_b = run_enrichr_generic(b_syms, library, tag="bhb")
    if df_d.empty or df_b.empty:
        return None

    overlaps = build_overlap_table(df_d, df_b, p_thresh=float(p_thresh))
    overlaps = annotate_terms_df(overlaps, is_overlap=True)
    disease_terms = prepare_disease_only_terms(df_d, df_b, p_thresh)
    st.session_state.disease_terms_all = disease_terms

    return overlaps

# ------------------------ Run / Reset controls ---------------------------
st.markdown("---")
with st.container():
    cA, cB, cC = st.columns([2,2,4])
    cA.info(f"Active disease set: **{len(st.session_state.disease_symbols)}** genes")
    cB.info(f"Active BHB set: **{len(st.session_state.bhb_symbols)}** genes")
    cC.caption(f"Sources → disease: {st.session_state.disease_source} | bhb: {st.session_state.bhb_source}")

run_disabled = (len(st.session_state.disease_symbols) == 0 or len(st.session_state.bhb_symbols) == 0)
run_clicked = st.button("▶️ Run enrichment & show overlaps", type="primary", disabled=run_disabled)
reset_clicked = st.button("Reset results")

if reset_clicked:
    st.session_state.results_ready = False
    st.session_state.overlaps_raw = None
    st.session_state.per_term_selected = {}
    st.session_state.term_open = {}
    st.session_state.term_heatmap_images = {}
    st.session_state.term_force_open = None
    st.session_state.scroll_target = None
    st.session_state.tissue_filter = []
    st.session_state.disease_terms_all = pd.DataFrame()
    st.session_state.show_all_disease_terms = False

if run_clicked:
    try:
        with st.spinner("Running Enrichr enrichment…"):
            res = _compute_and_store_results(
                st.session_state.disease_symbols,
                st.session_state.bhb_symbols,
                LIBRARY,
                P_THRESH
            )
        if res is None or res.empty:
            st.session_state.results_ready = False
            st.session_state.overlaps_raw = None
            st.session_state.per_term_selected = {}
            st.session_state.term_open = {}
            st.session_state.term_heatmap_images = {}
            st.session_state.term_force_open = None
            st.session_state.scroll_target = None
            st.session_state.tissue_filter = []
            st.session_state.disease_terms_all = pd.DataFrame()
            st.session_state.show_all_disease_terms = False
            st.warning("No overlapping terms at the selected threshold. Try adjusting your gene lists or p-value threshold.")
        else:
            st.session_state.results_ready = True
            st.session_state.overlaps_raw = res
            st.session_state.per_term_selected = {}
            st.session_state.term_open = {}
            st.session_state.term_heatmap_images = {}
            st.session_state.term_force_open = None
            st.session_state.scroll_target = None
            st.session_state.tissue_filter = []
            st.session_state.show_all_disease_terms = False
    except Exception as e:
        st.session_state.results_ready = False
        st.session_state.overlaps_raw = None
        st.session_state.per_term_selected = {}
        st.session_state.term_open = {}
        st.session_state.term_heatmap_images = {}
        st.session_state.term_force_open = None
        st.session_state.scroll_target = None
        st.session_state.tissue_filter = []
        st.session_state.disease_terms_all = pd.DataFrame()
        st.session_state.show_all_disease_terms = False
        st.error(f"Error during enrichment: {e}")

# --------- Inline renderer: clickable chips (buttons) + inline evidence --
def term_key_from_row(row) -> str:
    tid = getattr(row, "term_stable_id", None) or ""
    if tid:
        return re.sub(r"[^A-Za-z0-9_]+", "_", tid)
    nm = getattr(row, "term_name_clean", None) or getattr(row, "term", "")
    return re.sub(r"[^A-Za-z0-9_]+", "_", nm) or f"t_{hash(nm) & 0xffff}"

def render_bhb_gene_buttons(annotated_str: str, term_key: str):
    """
    Render BHB overlap genes as a grid of Streamlit buttons (chips).
    Clicking a chip selects the gene, marks term open, and forces rerun (prevents collapse).
    """
    if annotated_str is None or (isinstance(annotated_str, float) and np.isnan(annotated_str)):
        st.write("")
        return

    items = [t for t in str(annotated_str).split(";") if t.strip()]
    if not items:
        st.write("")
        return

    N_COLS = 6
    cols = st.columns(N_COLS, gap="small")

    for i, raw in enumerate(items):
        label = raw.strip()                       # keep * / **
        base = label.replace("*","").strip().upper()
        info = bhb_evidence.get(base, {})
        hint = f"GNN Score: {format_score(info.get('gnn_score'))} • Evidence: {format_int(info.get('evidence_count'))}"
        col = cols[i % N_COLS]
        if col.button(label, key=f"{term_key}_chip_{i}_{base}", help=hint):
            selected_map = dict(st.session_state.get("per_term_selected", {}))
            selected_map[term_key] = base
            st.session_state.per_term_selected = selected_map

            term_open_map = dict(st.session_state.get("term_open", {}))
            term_open_map[term_key] = True
            st.session_state.term_open = term_open_map     # keep this term open

            st.session_state.term_force_open = term_key
            st.session_state.scroll_target = term_key
            st.rerun()

def render_inline_gene_evidence(gene_symbol: str):
    """Inline evidence block: metrics + titles → PubMed + abstracts."""
    g = (gene_symbol or "").upper().strip()
    info = bhb_evidence.get(g, {})
    gnn = format_score(info.get("gnn_score"))
    evid = format_int(info.get("evidence_count"))
    pmids = info.get("pmids") or []
    titles = info.get("titles") or []
    abstracts = info.get("abstracts") or []

    st.markdown(f"### BHB-responsive gene — `{g}`")
    c1, c2 = st.columns(2)
    c1.metric("Score", gnn)
    c1.caption("*This score (0–1) reflects the strength of evidence that BHB influences this gene, combining study counts and network position.*")
    c2.metric("Number of supporting studies", evid)
    st.markdown("---")
    if not pmids and not titles and not abstracts:
        st.info("No publication details available for this gene in the BHB seeds table.")
        return
    n = max(len(pmids), len(titles), len(abstracts))
    for i in range(n):
        pmid = pmids[i] if i < len(pmids) else None
        ttl = titles[i] if i < len(titles) else "(untitled)"
        abs_ = abstracts[i] if i < len(abstracts) else ""
        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
        if link:
            st.markdown(f"**[{ttl}]({link})**")
        else:
            st.markdown(f"**{ttl}**")
        if abs_:
            st.write(abs_)
        st.markdown("---")

# -------------------------- Results section ------------------------------
if st.session_state.results_ready and st.session_state.overlaps_raw is not None:
    overlaps_df = st.session_state.overlaps_raw.copy()
    if overlaps_df is None:
        overlaps_df = pd.DataFrame()
    disease_terms_all = st.session_state.get("disease_terms_all")
    if disease_terms_all is None:
        disease_terms_all = pd.DataFrame()

    show_all = st.session_state.get("show_all_disease_terms", False)

    st.subheader("Reactome term results")
    if show_all:
        st.caption("Showing all Reactome terms enriched for the disease gene set (including overlaps with BHB-responsive genes).")
    else:
        st.caption("Terms where **both** gene sets have raw p-value below the selected threshold.")
    st.markdown("**Color coding – explanation**")
    st.markdown("🟢 Reactome terms overlap between disease genes and BHB-responsive genes.")
    st.markdown("🔵 Reactome terms are enriched only for the disease gene set (no significant BHB enrichment at the selected significance threshold in the left menu).")

    col_btn_show, col_btn_hide = st.columns([2, 2])
    if col_btn_show.button(
        "Show all Reactome terms enriched with Disease Genes",
        disabled=show_all or disease_terms_all.empty
    ):
        st.session_state.show_all_disease_terms = True
        st.rerun()
    if col_btn_hide.button(
        "Show only overlapping Reactome terms",
        disabled=not show_all
    ):
        st.session_state.show_all_disease_terms = False
        st.rerun()

    combined_df = overlaps_df.copy()
    if show_all and not disease_terms_all.empty:
        combined_df = pd.concat([combined_df, disease_terms_all.copy()], ignore_index=True, sort=False)
        if "is_overlap" in combined_df.columns:
            combined_df = combined_df.sort_values(["is_overlap"], ascending=False)
        dedup_cols = [c for c in ["term_stable_id", "_id_key", "_name_key"] if c in combined_df.columns]
        if dedup_cols:
            combined_df = combined_df.drop_duplicates(subset=dedup_cols, keep="first")

    if combined_df.empty:
        st.info("No Reactome terms available with the current selections.")
        st.stop()

    overlap_mask = combined_df.get("is_overlap")
    if overlap_mask is not None:
        overlap_count = int(pd.Series(overlap_mask).fillna(False).astype(bool).sum())
    else:
        overlap_count = combined_df.shape[0]
    disease_only_count = int(combined_df.shape[0] - overlap_count)
    if show_all:
        st.write(f"Displaying **{combined_df.shape[0]}** terms ({overlap_count} overlap with BHB-responsive genes, {disease_only_count} disease-only).")
    else:
        st.write(f"Found **{overlap_count}** overlapping terms at p < {P_THRESH}.")

    working_df = combined_df.copy()

    # ---- Size filter slider ----
    if working_df["n_genes"].notna().any():
        values = pd.to_numeric(working_df["n_genes"], errors="coerce").dropna().astype(int)
        cur_min, cur_max = int(values.min()), int(values.max())
        if cur_min < cur_max:
            default_hi = min(300, cur_max)
            size_range = st.slider(
                "Gene list size range (n_genes)",
                min_value=cur_min,
                max_value=cur_max,
                value=(cur_min, default_hi),
                help="Most actionable insights are often generated from lists with up to 300 genes."
            )
            working_df = working_df[
                working_df["n_genes"].fillna(cur_max + 1).between(size_range[0], size_range[1])
            ]
    else:
        st.info("Term sizes unavailable or single-valued; size filter disabled.")

    # ---- Tissue filter ----
    available_tissues = sorted({
        t.strip()
        for val in working_df["enriched_tissues"]
        if isinstance(val, str)
        for t in val.split(";")
        if t.strip()
    })
    with st.expander("Filter by enriched tissues", expanded=bool(st.session_state.get("tissue_filter"))):
        if available_tissues:
            selected_tissues = st.multiselect(
                "Show only Reactome terms enriched in these tissues:",
                options=available_tissues,
                default=[t for t in st.session_state.get("tissue_filter", []) if t in available_tissues],
                help="Reactome terms must be enriched in at least one of the selected tissues to be shown."
            )
            st.session_state.tissue_filter = selected_tissues
        else:
            st.info("No enriched tissues available to filter.")

    active_tissues = st.session_state.get("tissue_filter", [])
    if active_tissues:
        mask = working_df["enriched_tissues"].apply(
            lambda s: any(t in (s or "") for t in active_tissues)
        )
        working_df = working_df[mask]

    if working_df.empty:
        st.info("No Reactome terms match the current filters.")
        st.stop()

    # ---- Sorting options ----
    if show_all:
        working_df = working_df.sort_values(
            ["p_value_disease", "avg_p_value"],
            ascending=[True, True],
            na_position="last"
        )
    else:
        sort_by = st.selectbox(
            "Sort overlapping terms by",
            options=[
                "Average p-value (ascending)",
                "Gene list size (ascending)",
                "Gene list size (descending)",
                "BHB overlap (descending)",
                "BHB hub overlap (descending)",
            ],
            index=0
        )
        if sort_by == "Average p-value (ascending)":
            working_df = working_df.sort_values(
                ["avg_p_value","p_value_disease","p_value_bhb"],
                ascending=[True, True, True],
                na_position="last"
            )
        elif sort_by == "Gene list size (ascending)":
            working_df = working_df.sort_values(
                ["n_genes","avg_p_value"],
                ascending=[True, True],
                na_position="last"
            )
        elif sort_by == "Gene list size (descending)":
            working_df = working_df.sort_values(
                ["n_genes","avg_p_value"],
                ascending=[False, True],
                na_position="last"
            )
        elif sort_by == "BHB overlap (descending)":
            working_df = working_df.sort_values(
                ["overlap_count_bhb","avg_p_value"],
                ascending=[False, True],
                na_position="last"
            )
        else:  # "BHB hub overlap (descending)"
            working_df = working_df.sort_values(
                ["bhb_hub_overlap","avg_p_value"],
                ascending=[False, True],
                na_position="last"
            )

    # ---- Download filtered/sorted table ----
    csv_bytes = working_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download overlap CSV", data=csv_bytes, file_name="reactome_overlap_table.csv", mime="text/csv")

    if not MATPLOTLIB_AVAILABLE:
        st.warning("Install matplotlib (e.g., `pip install matplotlib`) to enable tissue expression heatmaps.")

    force_term_key = st.session_state.pop("term_force_open", None)
    scroll_target = st.session_state.pop("scroll_target", None)
    scroll_anchor_id = None

    # ---- Expanders: header + metrics + BHB gene chips + inline evidence ---
    for row in working_df.itertuples(index=False):
        size_str = "n/a" if pd.isna(getattr(row, "n_genes", np.nan)) else str(int(getattr(row, "n_genes")))
        avgp_str = "n/a" if pd.isna(getattr(row, "avg_p_value", np.nan)) else f"{getattr(row, 'avg_p_value'):.2e}"

        bhb_ov = getattr(row, "overlap_count_bhb", np.nan)
        bhb_hub = getattr(row, "bhb_hub_overlap", np.nan)
        bhb_ov_str  = "n/a" if pd.isna(bhb_ov)  else str(int(bhb_ov))
        bhb_hub_str = "n/a" if pd.isna(bhb_hub) else str(int(bhb_hub))

        title_name = getattr(row, "term_name_clean", None) or getattr(row, "term")
        title_id   = getattr(row, "term_stable_id", None)

        # Visible header with user-friendly descriptors
        gene_label = f"Gene set size={size_str}"
        bhb_label = f"BHB-responsive genes in this term: {bhb_ov_str}"
        hub_label = f"Hub genes responsive to BHB: {bhb_hub_str}"
        if title_id:
            header_description = f"{title_name} — {gene_label}, avg p={avgp_str}, {bhb_label}, {hub_label}"
        else:
            header_description = f"{title_name} — {gene_label}, avg p={avgp_str}, {bhb_label}, {hub_label}"

        prefix = "🟢" if getattr(row, "is_overlap", False) else "🔵"
        header_visible = f"{prefix} {header_description}"

        term_key = term_key_from_row(row)
        anchor_id = f"anchor_{term_key}"
        st.markdown(f"<span id='{anchor_id}'></span>", unsafe_allow_html=True)
        if scroll_target == term_key:
            scroll_anchor_id = anchor_id
        if force_term_key == term_key:
            st.session_state.term_open[term_key] = True
        st.session_state.term_open.setdefault(term_key, False)
        header_unique = f"{header_visible}\u200b{term_key}"

        sel_map = st.session_state.get("per_term_selected", {})
        has_selection = sel_map.get(term_key) is not None
        if has_selection:
            st.session_state.term_open[term_key] = True

        expanded_flag = st.session_state.term_open.get(term_key, False) or has_selection

        with st.expander(header_unique, expanded=expanded_flag):
            # Hierarchy row
            hierarchy = _format_parent_paths(getattr(row, "parent_paths", None))
            if hierarchy:
                st.markdown(f"**Hierarchy:** {hierarchy}")

            # --- Metrics grid: neat 2×4 layout (no zig-zag) ---
            t1, t2, t3, t4 = st.columns(4, gap="small")
            t1.metric("Disease enrichment strength (p-value)", f"{row.p_value_disease:.2e}" if pd.notna(row.p_value_disease) else "NA")
            t1.caption("*Disease-associated gene set vs this Reactome term; lower suggests stronger enrichment.*")
            t2.metric("Disease overlap", int(getattr(row, "overlap_count_disease", 0)))
            t2.caption("*Count of genes in the disease gene set that are members of this Reactome term.*")
            t3.metric("BHB enrichment strength (p-value)", f"{row.p_value_bhb:.2e}" if pd.notna(row.p_value_bhb) else "NA")
            t3.caption("*Set of BHB-responsive genes vs this Reactome term; lower suggests stronger enrichment.*")
            t4.metric("BHB overlap", int(getattr(row, "overlap_count_bhb", 0)))
            t4.caption("*Count of genes in the set of BHB-responsive genes that are members of this Reactome term.*")

            b1, b2, b3, b4 = st.columns(4, gap="small")
            b1.metric("Hub-Disease overlap", int(getattr(row, "disease_hub_overlap", 0)))
            b1.caption("*Hub genes in this term that are also in the disease gene set.*")
            b2.metric("Hub-BHB overlap", int(getattr(row, "bhb_hub_overlap", 0)))
            b2.caption("*Hub genes in this term that are also in the set of BHB-responsive genes.*")
            b3.metric("Overlapping hubs", int(getattr(row, "overlapping_hubs", 0)))
            b3.caption("*Number of hub genes present in both BHB-responsive genes and the disease gene set.*")
            avgp_show = f"{getattr(row, 'avg_p_value'):.2e}" if pd.notna(getattr(row, "avg_p_value", np.nan)) else "NA"
            b4.metric("Average p-value", avgp_show)

            # Hub genes in this term (annotated)
            hub_list = getattr(row, "hub_genes_term_annot", "")
            if hub_list:
                st.markdown("**Hub genes in this Reactome term:**")
                st.code(hub_list, language="text")

            tissue_list = getattr(row, "enriched_tissues", "")
            if isinstance(tissue_list, float) and pd.isna(tissue_list):
                tissue_list = ""
            if tissue_list:
                st.markdown("**Enriched in following tissues:**")
                st.code(str(tissue_list), language="text")
            else:
                st.markdown("**Enriched in following tissues:**")
                st.info("No tissues met the enrichment criteria (fold change > 0).")

            heatmap_store = dict(st.session_state.get("term_heatmap_images", {}))
            if st.button("Tissue Expression Heatmap", key=f"{term_key}_heatmap", disabled=not MATPLOTLIB_AVAILABLE):
                with st.spinner("Rendering tissue expression heatmap…"):
                    try:
                        png_bytes = build_term_heatmap_png(
                            getattr(row, "term_stable_id", None),
                            getattr(row, "_name_key", None),
                            title_name,
                        )
                        heatmap_store[term_key] = png_bytes
                        st.session_state.term_heatmap_images = heatmap_store
                        st.session_state.term_open[term_key] = True
                        st.session_state.term_force_open = term_key
                        st.session_state.scroll_target = term_key
                        st.rerun()
                    except Exception as e:
                        st.error(f"Heatmap error: {e}")

            stored_png = st.session_state.get("term_heatmap_images", {}).get(term_key)
            if stored_png:
                st.image(
                    stored_png,
                    caption=f"GTEx tissue expression heatmap — {title_name}",
                    use_container_width=True
                )

            # Overlap genes (Disease side)
            st.markdown("**Genes overlapping between this Reactome term and the disease gene set:**")
            st.code(str(getattr(row, "overlap_genes_disease_annot", getattr(row, "overlap_genes_disease", ""))), language="text")

            # Overlap genes (BHB side) — chips as buttons (no navigation)
            st.markdown("**Genes overlapping between this Reactome term and BHB-responsive genes:**  \nClick a gene to view studies showing that this gene is responsive to BHB.")
            render_bhb_gene_buttons(
                str(getattr(row, "overlap_genes_bhb_annot", getattr(row, "overlap_genes_bhb", ""))),
                term_key
            )

            # Inline evidence block for the currently selected gene in this term
            selected_gene = sel_map.get(term_key)
            if selected_gene:
                with st.expander(f"BHB-responsive gene — {selected_gene}", expanded=True):
                    render_inline_gene_evidence(selected_gene)
                    cc1, cc2 = st.columns([1,4])
                    if cc1.button("Hide", key=f"{term_key}_hide"):
                        # Clear just this term's selection; keep expander open
                        st.session_state.per_term_selected.pop(term_key, None)
                        st.session_state.term_open[term_key] = True
                        st.rerun()

    if scroll_anchor_id:
        st.markdown(
            f"<script>const el = document.getElementById('{scroll_anchor_id}'); if (el) {{ el.scrollIntoView({{behavior: 'auto', block: 'start'}}); }}</script>",
            unsafe_allow_html=True
        )

    # ---------------------- Optional: quick preview if you ship an example ----
    st.markdown("---")
with st.expander("🔎 Quick preview using the example precomputed overlap (no API calls)"):
    try:
        df_example = pd.read_csv("data/example_overlap_table.csv")
        st.dataframe(df_example.head(30), use_container_width=True)
        st.download_button(
            "Download example overlap CSV",
            data=df_example.to_csv(index=False).encode("utf-8"),
            file_name="example_overlap_table.csv",
            mime="text/csv"
        )
    except Exception:
        st.info("Example overlap not available.")
