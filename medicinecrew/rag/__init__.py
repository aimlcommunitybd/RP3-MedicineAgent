from crewai_tools import RAGAgent

# Add RAG tool for local drug database
rag_tool = RAGAgent(
    knowledge_base=...,
    document_path="dataset/db_drug_interactions.csv"
)