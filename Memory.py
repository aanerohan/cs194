import chromadb

# Initialize the ChromaDB client.
# 'PersistentClient' saves the data to disk in the specified path.
client = chromadb.PersistentClient(path="./agent_memory_db")

# Create a 'collection' which is like a table in a traditional database.
# If the collection already exists, it will load the existing one.
collection = client.get_or_create_collection(name="research_and_development")


def store_memory(doc_id: str, document: str, metadata: dict):
    """
    Stores a piece of text (a 'memory') in the vector database.
    'upsert' will add the document if it's new or update it if the id already exists.
    """
    collection.upsert(
        documents=[document],
        metadatas=[metadata],
        ids=[doc_id]
    )
    print(f"Memory stored with ID: {doc_id}")


def retrieve_memories(query_text: str, n_results: int = 2) -> list:
    """
    Queries the database to find the most relevant memories based on the query text.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results['documents'][0]
