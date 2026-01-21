import lancedb
from sentence_transformers import SentenceTransformer
from functools import lru_cache

# ---------- singleton model ----------
@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

# ---------- single reusable DB handle ----------
_db = lancedb.connect("src/rag/lancedb")
_table = _db.open_table("drug_interactions")

def search_interactions(query: str, limit: int = 3):
    """
    Fast vector search that lets LanceDB embed the query internally.
    Returns list[dict] with keys Drug 1, Drug 2, Interaction Description.
    """
    # LanceDB >= 0.5 accepts raw text and embeds under the hood
    df = (_table
          .search(query)            # text → embedding inside LanceDB
          .limit(limit)
          .to_pandas())            # deprecated to_df() removed

    return [{"Drug 1": r["drug1"],
             "Drug 2": r["drug2"],
             "Interaction Description": r["description"]}
            for _, r in df.iterrows()]

# ---------- quick self-test ----------
if __name__ == "__main__":
    print(search_interactions("napa and Ibuprofen"))
