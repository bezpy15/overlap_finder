import requests, pandas as pd

API_URL = "https://api.platform.opentargets.org/api/v4/graphql"

QUERY = """
query Targets($id: String!, $i: Int!, $n: Int!) {
  disease(efoId: $id) {
    name
    associatedTargets(page: {index: $i, size: $n}, orderByScore: "score") {
      count
      rows {
        target { id approvedSymbol approvedName }
        score
        datatypeScores   { id score }
        datasourceScores { id score }
      }
    }
  }
}
"""

def fetch_targets(disease_id: str, any_ot_score: float = 0.5) -> pd.DataFrame:
    def fetch_page(idx):
        res = requests.post(API_URL,
            json={"query": QUERY, "variables": {"id": disease_id, "i": idx, "n": 1000}},
            timeout=30).json()
        if "errors" in res:
            raise RuntimeError(res["errors"])
        return res["data"]["disease"]

    first = fetch_page(0)
    rows, total = first["associatedTargets"]["rows"], first["associatedTargets"]["count"]
    pages = (total + 999)//1000
    for p in range(1, pages):
        rows.extend(fetch_page(p)["associatedTargets"]["rows"])

    records = []
    for r in rows:
        rec = {"ensembl_id": r["target"]["id"],
               "symbol":     r["target"]["approvedSymbol"],
               "name":       r["target"]["approvedName"],
               "overall":    r["score"]}
        for c in r.get("datatypeScores", []):   rec[f"dt_{c['id']}"] = c["score"]
        for c in r.get("datasourceScores", []): rec[f"ds_{c['id']}"] = c["score"]
        records.append(rec)
    df = pd.DataFrame(records).dropna(subset=['symbol']).copy()
    score_cols = [c for c in df if c == "overall" or c.startswith(("dt_","ds_"))]
    df["max_ot_score"] = df[score_cols].max(axis=1, skipna=True)
    df = df[df["max_ot_score"] >= any_ot_score].copy()
    return df[["symbol","overall"] + [c for c in df.columns if c.startswith(("dt_","ds_"))]]
