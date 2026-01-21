import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

def create_database():
    # Load the dataset
    df = pd.read_csv("dataset/db_drug_interactions.csv")

    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create a combined column for embedding
    df['combined_drugs'] = df['Drug 1'] + " " + df['Drug 2']

    # Generate embeddings
    embeddings = model.encode(df['combined_drugs'].tolist(), show_progress_bar=True)

    # Create a LanceDB table
    db = lancedb.connect("src/rag/lancedb")
    table_name = "drug_interactions"
    
    # Prepare data for LanceDB
    data = []
    for i, row in df.iterrows():
        data.append({
            "vector": embeddings[i],
            "drug1": row["Drug 1"],
            "drug2": row["Drug 2"],
            "description": row["Interaction Description"]
        })

    # Create table
    try:
        db.drop_table(table_name)
        print(f"Table '{table_name}' dropped successfully.")
    except Exception:
        print(f"Table '{table_name}' does not exist, creating a new one.")

    table = db.create_table(table_name, data=data)
    print(f"Table '{table_name}' created successfully.")

    # build approximate index for sub-50 ms search
    print("Building index...")
    table.create_index(metric="L2", num_partitions=256, num_sub_vectors=96)
    print("Index created successfully.")


if __name__ == "__main__":
    create_database()
