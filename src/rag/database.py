import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path

def ingest_data():
    """
    Reads drug interaction data from a CSV file, creates embeddings,
    and stores them in a LanceDB table.
    """
    # Define the path to the database directory relative to this script
    db_path = Path(__file__).parent / "lancedb_data"

    # Connect to LanceDB
    db = lancedb.connect(db_path)
    
    # Load the dataset
    df = pd.read_csv("dataset/db_drug_interactions.csv")
    
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings for the drug names
    df["vector"] = df["drug_name"].apply(lambda x: model.encode(x))
    
    # Create a LanceDB table
    table_name = "drug_interactions"
    try:
        db.create_table(table_name, data=df)
        print(f"Table '{table_name}' created successfully.")
    except Exception as e:
        print(f"Table '{table_name}' already exists or an error occurred: {e}")

if __name__ == "__main__":
    ingest_data()
