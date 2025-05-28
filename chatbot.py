# -*- coding: utf-8 -*-
"""Intelligent Chatbot using Vector Database

This chatbot uses Zilliz Cloud for vector storage and Together AI for embeddings and LLM.
"""

import os
from typing import List, Dict, Any
from pymilvus import MilvusClient, DataType
from langchain_together import TogetherEmbeddings
from langchain_community.llms import Together
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain.schema import Document
import streamlit as st
import requests

# Configuration
CLUSTER_ENDPOINT = st.secrets["ZILLIZ_CLUSTER_ENDPOINT"]
TOKEN = st.secrets["ZILLIZ_TOKEN"]
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
COLLECTION_NAME = "chatbot_collection"
EMBEDDING_MODEL = "togethercomputer/m2-bert-80M-8k-retrieval"  # Updated model name
LLM_MODEL = "deepseek-ai/DeepSeek-V3"

# Validate required secrets
if not all([CLUSTER_ENDPOINT, TOKEN, TOGETHER_API_KEY]):
    st.error("Missing required secrets. Please check your Streamlit secrets configuration.")
    st.stop()

class VectorDatabase:
    def __init__(self):
        self.client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)
        self._setup_collection()
        
    def _setup_collection(self):
        """Initialize the vector database collection with proper schema"""
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
            print(f"Collection {COLLECTION_NAME} created successfully!")
        except Exception as e:
            print(f"Collection might already exist: {str(e)}")

class Chatbot:
    def __init__(self):
        print("Initializing Chatbot...")
        self.vector_db = VectorDatabase()
        print("Vector database initialized")
        
        print("Initializing embeddings...")
        try:
            # Test Together AI API directly
            print("Testing Together AI API...")
            headers = {
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            }

            data = {
                "model": EMBEDDING_MODEL,
                "input": ["test"]
            }

            response = requests.post("https://api.together.xyz/v1/embeddings", headers=headers, json=data)

            if response.status_code == 200:
                embedding = response.json()["data"][0]["embedding"]
                print("Embedding retrieved successfully. Length:", len(embedding))
            else:
                print("Failed to get embedding:", response.status_code, response.text)
                raise Exception(f"Failed to get embedding: {response.status_code} {response.text}")
            
            self.embeddings = TogetherEmbeddings(
                model=EMBEDDING_MODEL,
                api_key=TOGETHER_API_KEY
            )
            # Test the embeddings
            test_embedding = self.embeddings.embed_query("test")
            print(f"Embeddings initialized successfully. Test embedding dimension: {len(test_embedding)}")
        except Exception as e:
            print(f"Error initializing embeddings: {str(e)}")
            raise
        
        print("Initializing LLM...")
        try:
            self.llm = Together(
                model=LLM_MODEL,
                together_api_key=TOGETHER_API_KEY
            )
            # Test the LLM
            test_response = self.llm.invoke("test")
            print("LLM initialized successfully")
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            raise
        
        self.vector_store = None
        print("Chatbot initialization complete")
        
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load documents from various file types"""
        print(f"Attempting to load documents from: {file_paths}")
        documents = []
        for file_path in file_paths:
            try:
                # Check if file exists
                if not os.path.exists(file_path):
                    print(f"Error: File {file_path} does not exist!")
                    continue
                
                print(f"Reading file: {file_path}")
                # Read the file directly with UTF-8 encoding
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.strip():
                    print(f"Warning: File {file_path} is empty!")
                    continue
                
                print(f"File content length: {len(content)} characters")
                
                # Create a Document object manually
                doc = Document(
                    page_content=content,
                    metadata={"source": file_path}
                )
                documents.append(doc)
                print(f"Successfully loaded {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
        
        print(f"Total documents loaded: {len(documents)}")
        return documents
    
    def process_documents(self, documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50):
        """Process documents into chunks and store in vector database"""
        print("Starting document processing...")
        if not documents:
            print("No documents to process!")
            return
            
        try:
            print("Creating text splitter...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            print("Splitting documents into chunks...")
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                print("Warning: No chunks were created from the documents!")
                return
            
            print(f"Created {len(chunks)} chunks from documents")
            
            print("Testing embeddings with a sample chunk...")
            sample_text = chunks[0].page_content[:100]  # First 100 chars of first chunk
            try:
                test_embedding = self.embeddings.embed_query(sample_text)
                print(f"Embedding test successful. Dimension: {len(test_embedding)}")
            except Exception as e:
                print(f"Error testing embeddings: {str(e)}")
                raise
            
            print("Creating vector store...")
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
            print(f"Successfully processed {len(chunks)} document chunks")
            
            # Verify vector store
            if self.vector_store:
                print("Verifying vector store...")
                test_query = "test query"
                results = self.vector_store.similarity_search(test_query, k=1)
                print(f"Vector store verification successful. Retrieved {len(results)} results")
            else:
                print("Warning: Vector store is None after creation!")
                
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise
    
    def generate_response(self, query: str, k: int = 5) -> str:
        """Generate a response based on the query using retrieved context"""
        print(f"Generating response for query: {query}")
        if not self.vector_store:
            print("Error: Vector store is not initialized!")
            return "Error: No documents have been processed yet. Please load and process documents first."
            
        try:
            # Retrieve relevant documents
            print("Retrieving relevant documents...")
            search_results = self.vector_store.similarity_search(query, k=k)
            print(f"Retrieved {len(search_results)} relevant documents")
            
            # Build context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in search_results])
            
            # Create prompt
            prompt = f"""Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
            
            # Generate response
            print("Generating response from LLM...")
            response = self.llm.invoke(prompt).content
            print("Response generated successfully")
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return f"Error generating response: {str(e)}"

def main():
    # Initialize chatbot
    chatbot = Chatbot()
    
    # Example usage
    print("Chatbot initialized. Type 'quit' to exit.")
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() == 'quit':
            break
            
        response = chatbot.generate_response(query)
        print("\nResponse:", response)

if __name__ == "__main__":
    main()

