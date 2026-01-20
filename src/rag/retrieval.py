import lancedb
from sentence_transformers import SentenceTransformer
from pathlib import Path

def search_drug_interactions(query: str, limit: int = 5):
    """
    Searches for drug interactions in the LanceDB table.

    Args:
        query: The drug name to search for.
        limit: The maximum number of results to return.

    Returns:
        A list of drug interaction information.
    """
    # Define the path to the database directory relative to this script
    db_path = Path(__file__).parent / "lancedb_data"

    # Connect to LanceDB
    db = lancedb.connect(db_path)
    
    # Open the table
    table = db.open_table("drug_interactions")
    
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create an embedding for the query
    query_vector = model.encode(query)
    
    # Search for similar vectors in the table
    results = table.search(query_vector).limit(limit).to_df()
    
    return results.to_dict(orient="records")
