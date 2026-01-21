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
    Fast vector search.
    Manually embeds the query because the table was created without an automatic embedding schema.
    Returns list[dict] with keys Drug 1, Drug 2, Interaction Description.
    """
    # 1. Get the model instance
    model = _get_model()
    
    # 2. Convert text query to vector (embedding)
    # This fixes the error by matching the manual embedding approach used in database.py
    query_vector = model.encode(query)

    # 3. Perform vector search using the generated embedding
    df = (_table
          .search(query_vector)     # Pass vector, NOT raw text
          .limit(limit)
          .to_pandas())

    return [{"Drug 1": r["drug1"],
             "Drug 2": r["drug2"],
             "Interaction Description": r["description"]}
            for _, r in df.iterrows()]

# ---------- quick self-test ----------
if __name__ == "__main__":
    print(search_interactions("napa and Ibuprofen"))
