import lancedb
import sys
import argparse
from sentence_transformers import SentenceTransformer
from functools import lru_cache

# ---------- Singleton Model ----------
@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Single Reusable DB Handle ----------
_db = lancedb.connect("src/rag/lancedb")
_table = _db.open_table("drug_interactions")

def search_interactions(query: str, limit: int = 3):
    """
    Finds drug interactions by manually embedding the query 
    and performing a vector search in LanceDB.
    """
    model = _get_model()
    
    # Convert text query to vector (embedding)
    query_vector = model.encode(query)

    # Perform vector search using the generated embedding
    df = (_table
          .search(query_vector) 
          .limit(limit)
          .to_pandas())

    return [{"Drug 1": r["drug1"],
             "Drug 2": r["drug2"],
             "Interaction Description": r["description"]}
            for _, r in df.iterrows()]

# ---------- Main Execution Block ----------
if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Search for drug interactions.")

    # 'nargs='+' collects all positional arguments into a list.
    # This ensures it works whether you use quotes "Drug A and B" or not Drug A and B
    parser.add_argument("query", nargs='+', help="The search query text")
    
    # Add the optional limit argument
    parser.add_argument("--limit", type=int, default=3, help="Number of results to return (default: 3)")

    # Parse arguments
    args = parser.parse_args()
    
    # Join the query parts back into a single string
    full_query = " ".join(args.query)

    print(f"🔎 Searching for: '{full_query}' with limit {args.limit}...")
    results = search_interactions(full_query, limit=args.limit)

    # Pretty print the output
    if results:
        for i, res in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Drugs: {res['Drug 1']} + {res['Drug 2']}")
            print(f"Interaction: {res['Interaction Description']}")
    else:
        print("No results found.")
