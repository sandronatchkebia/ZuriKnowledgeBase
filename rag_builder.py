from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.settings import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
import chromadb
import os
from dotenv import load_dotenv

load_dotenv(override=True)

DATA_DIR = "./papers"
CHROMA_DB_DIR = "./chroma_db"

def build_index():
    # Set up persistent ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = chroma_client.get_or_create_collection("llm_papers")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Set embedding model
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    # Load + chunk documents
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    parser = SemanticSplitterNodeParser(embed_model=Settings.embed_model, chunk_size=768)
    nodes = parser.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes, storage_context=storage_context)
    print("Successfully built and registered the index")
    return index

def add_new_document(filepath):
    try:
        print(f"[DEBUG] Starting indexing for: {filepath}")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = chroma_client.get_or_create_collection("llm_papers")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

        abs_path = os.path.abspath(filepath)
        print(f"[DEBUG] Absolute path: {abs_path}")

        documents = SimpleDirectoryReader(input_files=[abs_path]).load_data()
        print(f"[DEBUG] Loaded {len(documents)} documents")

        parser = SemanticSplitterNodeParser(embed_model=Settings.embed_model, chunk_size=768)
        nodes = parser.get_nodes_from_documents(documents)
        print(f"[DEBUG] Parsed into {len(nodes)} nodes")

        index = VectorStoreIndex(nodes, storage_context=storage_context)
        print(f"Added {filepath} with {len(nodes)} chunks")
        return f"Added {filepath} with {len(nodes)} chunks"
    except Exception as e:
        print(f"Error adding document: {e}")
        return f"Failed to add document: {e}"

def load_index():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = chroma_client.get_or_create_collection("llm_papers")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    return VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

if __name__ == "__main__":
    build_index()
