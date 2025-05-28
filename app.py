import streamlit as st
from chatbot import Chatbot
import time
import os
import traceback

# Set page config
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    .chat-message.bot {
        background-color: #f0f2f6;
    }
    .chat-message .avatar {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    with st.spinner('Initializing chatbot...'):
        try:
            # Check if about.txt exists
            if not os.path.exists("about.txt"):
                st.error("Error: about.txt file not found!")
                st.stop()
            
            # Initialize chatbot with progress tracking
            st.write("Step 1: Initializing chatbot...")
            chatbot = Chatbot()
            st.write("Step 2: Chatbot initialized successfully")
            
            # Load and process the about document
            st.write("Step 3: Loading documents...")
            docs = chatbot.load_documents(["about.txt"])
            if not docs:
                st.error("Error: No documents were loaded from about.txt!")
                st.stop()
            
            st.write(f"Step 4: Successfully loaded {len(docs)} documents")
            
            st.write("Step 5: Processing documents...")
            chatbot.process_documents(docs)
            st.success("Step 6: Successfully loaded and processed documents!")
            
            # Store the chatbot in session state
            st.session_state.chatbot = chatbot
            
        except Exception as e:
            st.error(f"Error during initialization: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")
            st.stop()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Title
st.title("🤖 AI Chatbot")
st.markdown("Ask me anything about the content in about.txt!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.generate_response(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    This chatbot uses:
    - Zilliz Cloud for vector storage
    - Together AI for embeddings and LLM
    - Streamlit for the web interface
    
    The chatbot is trained on the content from about.txt.
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun() 
