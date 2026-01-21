import lancedb
import sys
from sentence_transformers import SentenceTransformer
from functools import lru_cache

# ---------- Singleton Model ----------
# Loads the model once and keeps it in memory for faster subsequent searches
@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Single Reusable DB Handle ----------
# Connects to the local database folder
_db = lancedb.connect("src/rag/lancedb")
_table = _db.open_table("drug_interactions")

def search_interactions(query: str, limit: int = 3):
    """
    Finds drug interactions by manually embedding the query 
    and performing a vector search in LanceDB.
    """
    # 1. Get the model instance
    model = _get_model()
    
    # 2. Convert text query to vector (embedding)
    # This is required because the table was created without an automatic embedding schema.
    query_vector = model.encode(query)

    # 3. Perform vector search using the generated embedding
    # We pass the vector list, not the raw text string
    df = (_table
          .search(query_vector) 
          .limit(limit)
          .to_pandas())

    # 4. Format the results
    return [{"Drug 1": r["drug1"],
             "Drug 2": r["drug2"],
             "Interaction Description": r["description"]}
            for _, r in df.iterrows()]

# ---------- Main Execution Block ----------
if __name__ == "__main__":
    # check if arguments were passed via command line
    # sys.argv[0] is the script name, so we look for arguments after that
    if len(sys.argv) > 1:
        # Join all arguments to handle queries with spaces (e.g., "drug A and drug B")
        query = " ".join(sys.argv[1:])
    else:
        # Default query if nothing is typed
        query = "napa and Ibuprofen"

    print(f"🔎 Searching for: '{query}'...")
    results = search_interactions(query)

    # Pretty print the output
    if results:
        for i, res in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Drugs: {res['Drug 1']} + {res['Drug 2']}")
            print(f"Interaction: {res['Interaction Description']}")
    else:
        print("No results found.")
