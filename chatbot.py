import os
from typing import List, Dict, Any
from pymilvus import MilvusClient, DataType
from langchain_community.llms import Together
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import streamlit as st
import requests

# Configuration
CLUSTER_ENDPOINT = st.secrets["ZILLIZ_CLUSTER_ENDPOINT"]
TOKEN = st.secrets["ZILLIZ_TOKEN"]
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
COLLECTION_NAME = "chatbot_collection"
EMBEDDING_MODEL = "togethercomputer/m2-bert-80M-8k-retrieval"
LLM_MODEL = "deepseek-ai/DeepSeek-V3"

# Validate required secrets
if not all([CLUSTER_ENDPOINT, TOKEN, TOGETHER_API_KEY]):
    st.error("Missing required secrets. Please check your Streamlit secrets configuration.")
    st.stop()

class TogetherEmbeddings(Embeddings):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), 10):
            batch = texts[i:i+10]
            response = requests.post(
                "https://api.together.xyz/v1/embeddings",
                headers=self.headers,
                json={"model": self.model, "input": batch}
            )
            if response.status_code == 200:
                batch_embeddings = [item["embedding"] for item in response.json()["data"]]
                embeddings.extend(batch_embeddings)
            else:
                raise Exception(f"Failed to get embeddings: {response.status_code} {response.text}")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            "https://api.together.xyz/v1/embeddings",
            headers=self.headers,
            json={"model": self.model, "input": [text]}
        )
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            raise Exception(f"Failed to get embedding: {response.status_code} {response.text}")

class VectorDatabase:
    def __init__(self):
        self.client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)
        self._setup_collection()
        
    def _setup_collection(self):
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=True
        )
        
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        
        try:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=schema,
                index_params=self.client.prepare_index_params(
                    field_name="vector",
                    index_type="AUTOINDEX",
                    metric_type="L2"
                )
            )
        except Exception:
            pass

class Chatbot:
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.embeddings = TogetherEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=TOGETHER_API_KEY
        )
        self.llm = Together(
            model=LLM_MODEL,
            together_api_key=TOGETHER_API_KEY
        )
        self.vector_store = None
        
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        documents = []
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                doc = Document(
                    page_content=content,
                    metadata={"source": file_path}
                )
                documents.append(doc)
            except Exception as e:
                continue
        
        return documents
    
    def process_documents(self, documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50):
        if not documents:
            return
            
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                return
            
            self.vector_store = Milvus.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=COLLECTION_NAME,
                connection_args={"uri": CLUSTER_ENDPOINT, "token": TOKEN},
                primary_field="id",
                text_field="text",
                vector_field="vector",
                metadata_field="metadata",
                drop_old=False
            )
            
        except Exception as e:
            raise
    
    def generate_response(self, query: str, k: int = 5) -> str:
        if not self.vector_store:
            return "Error: No documents have been processed yet. Please load and process documents first."
            
        try:
            search_results = self.vector_store.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content for doc in search_results])
            prompt = f"""Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
            response = self.llm.invoke(prompt)
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

def main():
    chatbot = Chatbot()
    print("Chatbot initialized. Type 'quit' to exit.")
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() == 'quit':
            break
        response = chatbot.generate_response(query)
        print("\nResponse:", response)

if __name__ == "__main__":
    main()
