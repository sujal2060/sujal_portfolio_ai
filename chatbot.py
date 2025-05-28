# -*- coding: utf-8 -*-
"""Intelligent Chatbot using Vector Database

This chatbot uses Zilliz Cloud for vector storage and Together AI for embeddings and LLM.
"""

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
import time

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
    """Custom embeddings class for Together AI"""
    
    def __init__(self, model: str, api_key: str):
        print(f"Initializing TogetherEmbeddings with model: {model}")
        self.model = model
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        print("TogetherEmbeddings initialized")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        print(f"Embedding {len(texts)} documents...")
        embeddings = []
        # Process in batches of 10
        for i in range(0, len(texts), 10):
            batch = texts[i:i+10]
            print(f"Processing batch {i//10 + 1} of {(len(texts) + 9)//10}")
            start_time = time.time()
            response = requests.post(
                "https://api.together.xyz/v1/embeddings",
                headers=self.headers,
                json={"model": self.model, "input": batch}
            )
            if response.status_code == 200:
                batch_embeddings = [item["embedding"] for item in response.json()["data"]]
                embeddings.extend(batch_embeddings)
                print(f"Batch processed in {time.time() - start_time:.2f} seconds")
            else:
                print(f"Error response: {response.status_code} - {response.text}")
                raise Exception(f"Failed to get embeddings: {response.status_code} {response.text}")
        print(f"Successfully embedded {len(embeddings)} documents")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query"""
        print(f"Embedding query: {text[:50]}...")
        start_time = time.time()
        response = requests.post(
            "https://api.together.xyz/v1/embeddings",
            headers=self.headers,
            json={"model": self.model, "input": [text]}
        )
        if response.status_code == 200:
            embedding = response.json()["data"][0]["embedding"]
            print(f"Query embedded in {time.time() - start_time:.2f} seconds")
            return embedding
        else:
            print(f"Error response: {response.status_code} - {response.text}")
            raise Exception(f"Failed to get embedding: {response.status_code} {response.text}")

class VectorDatabase:
    def __init__(self):
        print("Initializing VectorDatabase...")
        self.client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)
        print("MilvusClient initialized")
        self._setup_collection()
        
    def _setup_collection(self):
        """Initialize the vector database collection with proper schema"""
        print("Setting up collection...")
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
            # Initialize custom embeddings
            self.embeddings = TogetherEmbeddings(
                model=EMBEDDING_MODEL,
                api_key=TOGETHER_API_KEY
            )
            
            # Test the embeddings
            print("Testing embeddings with a sample query...")
            test_embedding = self.embeddings.embed_query("test")
            print(f"Embeddings initialized successfully. Test embedding dimension: {len(test_embedding)}")
            
            print("Initializing LLM...")
            self.llm = Together(
                model=LLM_MODEL,
                together_api_key=TOGETHER_API_KEY
            )
            # Test the LLM
            print("Testing LLM with a sample query...")
            test_response = self.llm.invoke("test")
            print("LLM initialized successfully")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
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
                print(f"First 100 characters of content: {content[:100]}")
                
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
        if len(documents) == 0:
            print("Warning: No documents were loaded!")
        return documents
    
    def process_documents(self, documents: List[Document], chunk_size: int = 50, chunk_overlap: int = 25):
        """Process documents into chunks and store in vector database"""
        print("Starting document processing...")
        if not documents:
            print("No documents to process!")
            return
            
        try:
            print("Creating text splitter...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            print("Splitting documents into chunks...")
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                print("Warning: No chunks were created from the documents!")
                return
            
            print(f"Created {len(chunks)} chunks from documents")
            print("Sample of first chunk:", chunks[0].page_content[:100] if chunks else "No chunks")
            
            print("Creating vector store...")
            start_time = time.time()
            self.vector_store = Milvus.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=COLLECTION_NAME,
                connection_args={"uri": CLUSTER_ENDPOINT, "token": TOKEN},
                primary_field="id",
                text_field="text",
                vector_field="vector",
                metadata_field="metadata",
                drop_old=True  # Changed to True to ensure fresh data
            )
            print(f"Vector store created in {time.time() - start_time:.2f} seconds")
            print(f"Successfully processed {len(chunks)} document chunks")
            print("Document processing completed successfully!")
            
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise
    
    def generate_response(self, query: str, k: int = 15) -> str:
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
            print(f"Context length: {len(context)} characters")
            print(f"First 100 characters of context: {context[:100]}")
            
            # Create a more detailed prompt
            prompt = f"""You are an AI assistant trained on Sujal Devkota's personal information, projects, and blog posts. 
            Your task is to provide accurate and consistent information about Sujal based on the provided context.
            
            Important sections to focus on:
            - Personal Information (including family details)
            - Contact Information
            - Skills
            - About Me
            - What I Offer (Services and Pricing)
            - My Favorite Places
            - Blog Posts
            - Projects
            - Family Information (parents, siblings, etc.)

            Context:
            {context}

            Question: {query}

            Instructions:
            1. Answer based ONLY on the provided context about Sujal
            2. If the context doesn't contain the answer, say "I don't have enough information to answer that question"
            3. Be specific and detailed in your response
            4. If the question is unclear, ask for clarification
            5. Maintain a professional and helpful tone
            6. When discussing projects or blog posts, include relevant details and dates
            7. When discussing skills or services, be specific about what Sujal offers
            8. If asked about pricing, provide the exact amounts mentioned in the context
            9. For questions about who Sujal is, focus on the personal information and about me sections
            10. For questions about family members, carefully check the personal information section
            11. Pay special attention to any mentions of parents, siblings, or other family members
            12. If family information is mentioned, include it in your response
            13. Always verify information across multiple chunks to ensure consistency
            14. If you find conflicting information, use the most detailed or recent information
            15. For questions about identity or background, combine information from multiple sections

            Answer:"""
            
            # Generate response
            print("Generating response from LLM...")
            response = self.llm.invoke(prompt)
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

