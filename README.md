# AI Chatbot with Streamlit

This is an AI-powered chatbot built with Streamlit, using Zilliz Cloud for vector storage and Together AI for embeddings and LLM.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. For local development, create a `.streamlit/secrets.toml` file in your project directory with:

```toml
ZILLIZ_CLUSTER_ENDPOINT = "your_zilliz_endpoint"
ZILLIZ_TOKEN = "your_zilliz_token"
TOGETHER_API_KEY = "your_together_api_key"
```

3. Run the application locally:

```bash
streamlit run app.py
```

## Deployment Options

### Option 1: Streamlit Cloud

1. Push your code to a GitHub repository

2. Go to [Streamlit Cloud](https://streamlit.io/cloud)

3. Click "New app" and connect your GitHub repository

4. Set the following:

   - Main file path: `app.py`
   - Python version: 3.9 or higher

5. Add your secrets in the Streamlit Cloud dashboard:

   - Go to your app's settings
   - Click on "Secrets"
   - Add the following secrets:
     ```toml
     ZILLIZ_CLUSTER_ENDPOINT = "your_zilliz_endpoint"
     ZILLIZ_TOKEN = "your_zilliz_token"
     TOGETHER_API_KEY = "your_together_api_key"
     ```

6. Deploy!

### Option 2: Render

1. Push your code to a GitHub repository

2. Go to [Render Dashboard](https://dashboard.render.com)

3. Click "New +" and select "Web Service"

4. Connect your GitHub repository

5. Configure the service:

   - Name: `ai-chatbot` (or your preferred name)
   - Environment: `Python`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py`

6. Add your environment variables:

   - ZILLIZ_CLUSTER_ENDPOINT
   - ZILLIZ_TOKEN
   - TOGETHER_API_KEY

7. Click "Create Web Service"

## Files Structure

- `app.py`: Main Streamlit application
- `chatbot.py`: Chatbot implementation
- `about.txt`: Training data for the chatbot
- `requirements.txt`: Python dependencies
- `render.yaml`: Render deployment configuration
- `Procfile`: Process file for Render
