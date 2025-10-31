import re, time, requests, pandas as pd, numpy as np

ENRICHR_ADD_URL    = "https://maayanlab.cloud/Enrichr/addList"
ENRICHR_ENRICH_URL = "https://maayanlab.cloud/Enrichr/enrich"

def _ascii(s: str) -> str:
    return str(s).encode("ascii","ignore").decode("ascii")

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+","_", str(s)).strip("_").lower()

def _norm_term(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def run_enrichr_generic(gene_symbols, library: str, tag: str):
    genes = [g for g in pd.Series(gene_symbols).dropna().astype(str).unique() if g]
    if not genes:
        return pd.DataFrame()
    desc = _ascii(f"{tag}_geneset")

    def _try_add(method: str):
        if method == "files":
            files = {"list": ("genes.txt", "\n".join(genes))}
            data  = {"description": desc}
            r = requests.post(ENRICHR_ADD_URL, files=files, data=data, timeout=60)
        else:
            data = {"list": "\n".join(genes), "description": desc}
            r = requests.post(ENRICHR_ADD_URL, data=data, timeout=60)
        r.raise_for_status()
        j = r.json()
        if "userListId" not in j:
            raise RuntimeError(f"Enrichr addList response missing userListId: {j}")
        return j["userListId"]

    user_list_id, last_err = None, None
    for method in ("files", "data"):
        for attempt in range(3):
            try:
                user_list_id = _try_add(method); break
            except Exception as e:
                last_err = e; time.sleep(2*(attempt+1))
        if user_list_id: break
    if not user_list_id:
        raise RuntimeError(f"Enrichr addList failed after retries: {last_err}")

    payload = {}
    for attempt in range(3):
        try:
            r = requests.get(ENRICHR_ENRICH_URL, params={"userListId": user_list_id, "backgroundType": library}, timeout=120)
            r.raise_for_status()
            payload = r.json()
            break
        except Exception as e:
            if attempt == 2: raise RuntimeError(f"Enrichr enrich failed after retries: {e}")
            time.sleep(2*(attempt+1))

    raw = payload.get(library, [])
    rows = []
    for rec in raw:
        term           = rec[1] if len(rec) > 1 else None
        p_value        = rec[2] if len(rec) > 2 else None
        z_score        = rec[3] if len(rec) > 3 else None
        combined_score = rec[4] if len(rec) > 4 else None
        overlap_genes  = rec[5] if len(rec) > 5 else ""
        if isinstance(overlap_genes, list): overlap_genes = ";".join(overlap_genes)
        adj_p_value    = rec[6] if len(rec) > 6 else None
        rank           = rec[0] if len(rec) > 0 else None
        rows.append({
            "rank": rank,
            "term": term,
            "p_value": p_value,
            "z_score": z_score,
            "combined_score": combined_score,
            "overlap_genes": overlap_genes,
            "overlap_count": (0 if not overlap_genes else len([g for g in str(overlap_genes).split(";") if g])),
            "adj_p_value": adj_p_value,
            "-log10(p_value)": (None if (p_value is None or p_value <= 0) else -np.log10(p_value)),
            "-log10(adj_p_value)": (None if (adj_p_value is None or adj_p_value <= 0) else -np.log10(adj_p_value)),
            "library": library,
            "input_gene_count": len(genes),
        })
    df = pd.DataFrame(rows)
    for col in ["p_value","adj_p_value","z_score","combined_score","overlap_count"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    if not df.empty:
        df = df.sort_values(["adj_p_value","p_value"], ascending=[True, True], na_position="last").reset_index(drop=True)
    return df

def build_overlap_table(df_disease: pd.DataFrame, df_bhb: pd.DataFrame, p_thresh: float):
    d = df_disease.loc[df_disease["p_value"] < p_thresh, ["term","p_value","adj_p_value","overlap_count","overlap_genes"]].copy()
    b = df_bhb.loc[df_bhb["p_value"] < p_thresh, ["term","p_value","adj_p_value","overlap_count","overlap_genes"]].copy()

    d = d.rename(columns={
        "p_value":"p_value_disease",
        "adj_p_value":"adj_p_value_disease",
        "overlap_count":"overlap_count_disease",
        "overlap_genes":"overlap_genes_disease",
    })
    b = b.rename(columns={
        "p_value":"p_value_bhb",
        "adj_p_value":"adj_p_value_bhb",
        "overlap_count":"overlap_count_bhb",
        "overlap_genes":"overlap_genes_bhb",
    })

    d["_term_key"] = _norm_term(d["term"])
    b["_term_key"] = _norm_term(b["term"])

    merged = d.merge(b.drop(columns=["term"]), on="_term_key", how="inner").drop(columns=["_term_key"]).copy()

    # enforce BOTH raw p-values below threshold
    both_mask = (
        merged["p_value_disease"].notna() &
        merged["p_value_bhb"].notna() &
        (merged["p_value_disease"] < p_thresh) &
        (merged["p_value_bhb"]   < p_thresh)
    )
    merged = merged.loc[both_mask].copy()

    # basic ordering
    final = merged.sort_values(
        ["adj_p_value_disease","adj_p_value_bhb"], ascending=[True, True]
    ).reset_index(drop=True)
    return final
