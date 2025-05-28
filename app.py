import streamlit as st
from chatbot import Chatbot
import time
import os

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
            # Print current working directory and list files
            st.write(f"Current working directory: {os.getcwd()}")
            st.write("Files in directory:", os.listdir())
            
            st.session_state.chatbot = Chatbot()
            
            # Check if about.txt exists
            if not os.path.exists("about.txt"):
                st.error("Error: about.txt file not found in the current directory!")
                st.stop()
            
            st.write("Found about.txt file, attempting to load...")
            
            # Load and process the about document
            docs = st.session_state.chatbot.load_documents(["about.txt"])
            if not docs:
                st.error("Error: No documents were loaded from about.txt!")
                st.stop()
            
            st.write(f"Successfully loaded {len(docs)} documents")
            
            st.write("Processing documents...")
            st.session_state.chatbot.process_documents(docs)
            st.success("Successfully loaded and processed documents!")
            
        except Exception as e:
            st.error(f"Error during initialization: {str(e)}")
            import traceback
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