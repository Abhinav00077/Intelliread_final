# Intelliread: PDF Q&A with LangChain and Streamlit

This project is a Streamlit application that allows you to upload a PDF file and ask questions about its content. It uses LangChain, OpenAI, and Pinecone to provide intelligent answers.

## Getting Started

### Prerequisites

- Python 3.7+
- A GitHub account
- API keys for OpenAI and Pinecone

### Local Development

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd <repo-name>
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your secrets:**

    Create a file at `.streamlit/secrets.toml` and add your API keys:

    ```toml
    # .streamlit/secrets.toml
    OPENAI_API_KEY = "sk-..."
    PINECONE_API_KEY = "your-pinecone-api-key"
    PINECONE_ENVIRONMENT = "your-pinecone-environment" # e.g., "gcp-starter"
    ```

    Replace the placeholder values with your actual credentials.

4.  **Run the application:**

    ```bash
    streamlit run main.py
    ```

## Deploying to Streamlit Community Cloud

1.  **Push your code to GitHub:**

    Make sure your latest code, including `main.py`, `requirements.txt`, and the empty `README.md` are pushed to a public or private GitHub repository. The `.streamlit/secrets.toml` file should *not* be pushed to GitHub (it's in `.gitignore`).

2.  **Sign up for Streamlit Community Cloud:**

    If you don't have an account, sign up at [streamlit.io/cloud](https://streamlit.io/cloud).

3.  **Deploy your app:**

    - From your Streamlit Cloud dashboard, click "New app".
    - Connect your GitHub account.
    - Select the repository and branch you want to deploy.
    - The main file path should be `main.py`.
    - Click "Advanced settings".
    - In the "Secrets" section, paste the contents of your local `.streamlit/secrets.toml` file.

    ```toml
    OPENAI_API_KEY = "sk-..."
    PINECONE_API_KEY = "your-pinecone-api-key"
    PINECONE_ENVIRONMENT = "your-pinecone-environment"
    ```

4.  **Click "Deploy!"**

Your application will be deployed and accessible via a public URL.
