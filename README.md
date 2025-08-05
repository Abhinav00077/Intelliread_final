# Smart PDF Reader

![smart pdf reader](/screenshots/IMG_4161.JPG)

## Overview

The Smart PDF Reader is a comprehensive project that harnesses the power of the Retrieval-Augmented Generation (RAG) model over a Large Language Model (LLM) powered by Langchain. Additionally, it utilizes the Pinecone vector database to efficiently store and retrieve vectors associated with PDF documents. This approach enables the extraction of essential information from PDF files without the need for training the model on question-answering datasets.

## Features

1. **RAG Model Integration**: The project seamlessly integrates the Retrieval-Augmented Generation (RAG) model, combining a retriever and a generator for effective question answering.

2. **Langchain-powered Large Language Model (LLM)**: Langchain enhances the capabilities of the Large Language Model, providing advanced language understanding and context.

3. **Pinecone Vector Database**: Utilizing Pinecone's vector database allows for efficient storage and retrieval of document vectors, optimizing the overall performance of the Smart PDF Reader.

4. **PDF Information Extraction**: The system focuses on extracting information directly from PDF files, eliminating the need for extensive training on question answering datasets.

5. **User-Friendly Interface**: The project includes a user-friendly interface for interacting with the PDF reader, making it accessible to users with various levels of technical expertise.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Pinecone API key and environment

### Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd RAG-on-PDF
   ```

2. **Run the setup script**:
   ```bash
   python setup.py
   ```

3. **Set up your API keys**:
   - Edit the `.env` file created by the setup script
   - Add your OpenAI API key (get from https://platform.openai.com/api-keys)
   - Add your Pinecone API key and environment (get from https://app.pinecone.io/)

4. **Place a PDF file** in the project directory

5. **Run the application**:
   ```bash
   python main.py
   ```

### Manual Installation

If you prefer to install manually:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create a `.env` file** with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment_here
   ```

3. **Run the application**:
   ```bash
   python main.py
   ```

## Usage

1. **Start the application**: The system will automatically detect PDF files in the directory and process the first one found.

2. **Processing**: The system will:
   - Extract text from the PDF
   - Clean and preprocess the text
   - Split it into manageable chunks
   - Upload chunks to Pinecone vector database
   - Set up the question-answering system

3. **Interactive Q&A**: Once processing is complete, you can ask questions about the PDF content interactively.

4. **Exit**: Type 'quit', 'exit', or 'q' to exit the application.

## Project Structure

```
RAG-on-PDF/
├── main.py                 # Main application script
├── setup.py               # Setup script for easy installation
├── requirements.txt       # Python dependencies
├── source/
│   ├── extract_text.py    # PDF text extraction utilities
│   └── cleaning_pipeline.py # Text cleaning and preprocessing
├── notebooks/
│   └── pdfReader.ipynb    # Original Jupyter notebook implementation
├── screenshots/           # Project screenshots
└── README.md             # This file
```

## Dependencies

- **Core ML/AI**: torch, transformers, langchain, google-generativeai
- **Vector Database**: pinecone-client
- **PDF Processing**: PyMuPDF
- **Text Processing**: nltk, tiktoken
- **Utilities**: tqdm, python-dotenv, requests

## API Setup

### Gemini API
1. Go to https://makersuite.google.com/app/apikey
2. Create a new API key
3. Add it to your `.env` file as `GEMINI_API_KEY`

### Pinecone API
1. Go to https://app.pinecone.io/
2. Create an account and get your API key
3. Note your environment (e.g., 'us-east1-gcp')
4. Add both to your `.env` file as `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT`

## How It Works

1. **Text Extraction**: Uses PyMuPDF to extract text from PDF documents
2. **Text Cleaning**: Applies various filters to clean and normalize the extracted text
3. **Chunking**: Splits the text into sentence-based chunks for optimal processing
4. **Embedding**: Converts text chunks into vector embeddings using Gemini's embedding-001 model
5. **Vector Storage**: Stores embeddings in Pinecone vector database for efficient retrieval
6. **Question Answering**: Uses RAG (Retrieval-Augmented Generation) to answer questions by:
   - Finding relevant text chunks using vector similarity search
   - Generating answers using Gemini 2.0 Flash with retrieved context

## Troubleshooting

### Common Issues

1. **API Key Errors**: Make sure your API keys are correctly set in the `.env` file
2. **Rate Limiting**: The system includes rate limiting, but you may need to wait if you hit OpenAI's rate limits
3. **PDF Processing**: Ensure your PDF file is not corrupted and contains extractable text
4. **Memory Issues**: For large PDFs, consider processing smaller sections

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed correctly
2. Verify your API keys are valid and have sufficient credits
3. Ensure your PDF file is accessible and contains text

## License

This project is licensed under the [MIT License](LICENSE).

---

## Blog
Read about [Vector Database Architecture](https://arshad-kazi.com/vector-database-and-its-architecture/)
