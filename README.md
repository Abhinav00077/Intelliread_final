# Smart PDF Reader

![smart pdf reader](/screenshots/IMG_4161.JPG)

## Overview

The Smart PDF Reader is a comprehensive project that harnesses the power of the Retrieval-Augmented Generation (RAG) model using Google's Gemini API. It utilizes the Pinecone vector database to efficiently store and retrieve vectors associated with PDF documents. This approach enables the extraction of essential information from PDF files without the need for training the model on question-answering datasets.

## Features

1. **RAG Model Integration**: The project seamlessly integrates the Retrieval-Augmented Generation (RAG) model, combining a retriever and a generator for effective question answering.

2. **Google Gemini API Integration**: Direct integration with Google's Gemini API for advanced language understanding and context generation.

3. **Pinecone Vector Database**: Utilizing Pinecone's vector database allows for efficient storage and retrieval of document vectors, optimizing the overall performance of the Smart PDF Reader.

4. **Flexible Embedding System**: Supports both TF-IDF and neural embeddings (Sentence Transformers) for text vectorization.

5. **PDF Information Extraction**: The system focuses on extracting information directly from PDF files, eliminating the need for extensive training on question answering datasets.

6. **User-Friendly Interface**: The project includes a user-friendly interface for interacting with the PDF reader, making it accessible to users with various levels of technical expertise.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Pinecone API key and environment

### Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd Intelliread_final
   ```

2. **Run the setup script**:
   ```bash
   python setup.py
   ```

3. **Set up your API keys**:
   - Edit the `.env` file created by the setup script
   - Add your Google Gemini API key (get from https://makersuite.google.com/app/apikey)
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
   GEMINI_API_KEY=your_gemini_api_key_here
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
   - Create embeddings using TF-IDF (with fallback to neural embeddings)
   - Upload chunks to Pinecone vector database
   - Set up the question-answering system

3. **Interactive Q&A**: Once processing is complete, you can ask questions about the PDF content interactively.

4. **Exit**: Type 'quit', 'exit', or 'q' to exit the application.

## Project Structure

```
Intelliread_final/
├── main.py                 # Main application script
├── setup.py               # Setup script for easy installation
├── requirements.txt       # Python dependencies
├── source/
│   ├── extract_text.py    # PDF text extraction utilities
│   └── cleaning_pipeline.py # Text cleaning and preprocessing
├── screenshots/           # Project screenshots
└── README.md             # This file
```

## Dependencies

- **Core ML/AI**: google-generativeai
- **Vector Database**: pinecone-client>=5.0.0
- **PDF Processing**: PyMuPDF
- **Text Processing**: nltk, scikit-learn
- **Embeddings**: sentence-transformers (for neural embeddings)
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

## Using Local LLMs (Ollama)

You can also use local LLMs like Ollama Mistral instead of cloud APIs for enhanced privacy and offline capabilities.

### Setting up Ollama

1. **Install Ollama**:
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows
   # Download from https://ollama.ai/download
   ```

2. **Pull the Mistral model**:
   ```bash
   ollama pull mistral
   ```

3. **Start Ollama service**:
   ```bash
   ollama serve
   ```

### Modifying the Project for Local LLMs

1. **Install additional dependencies**:
   ```bash
   pip install ollama
   ```

2. **Create a local LLM configuration**:
   Create a new file `local_llm_config.py`:
   ```python
   import ollama
   
   class LocalLLM:
       def __init__(self, model_name="mistral"):
           self.model_name = model_name
           
       def generate_text(self, prompt, max_tokens=1000):
           try:
               response = ollama.chat(model=self.model_name, messages=[
                   {
                       'role': 'user',
                       'content': prompt
                   }
               ])
               return response['message']['content']
           except Exception as e:
               print(f"Error generating text: {e}")
               return None
   ```

3. **Update main.py to use local LLM**:
   ```python
   # Replace GeminiAPI with LocalLLM
   from local_llm_config import LocalLLM
   
   # In SmartPDFReader.__init__()
   self.llm = LocalLLM("mistral")  # or any other model
   ```

### Available Ollama Models

- **mistral**: Fast and efficient (recommended)
- **llama2**: Meta's Llama 2 model
- **codellama**: Specialized for code
- **neural-chat**: Intel's optimized model

### Benefits of Local LLMs

- **Privacy**: No data sent to external servers
- **Offline**: Works without internet connection
- **Cost**: No API usage fees
- **Customization**: Can fine-tune models for specific use cases
- **Speed**: Lower latency for local processing

### Performance Considerations

- **Hardware Requirements**: Local LLMs require more RAM and GPU resources
- **Model Size**: Larger models provide better results but need more resources
- **Inference Speed**: Local processing may be slower than cloud APIs depending on hardware

## How It Works

1. **Text Extraction**: Uses PyMuPDF to extract text from PDF documents
2. **Text Cleaning**: Applies various filters to clean and normalize the extracted text
3. **Chunking**: Splits the text into sentence-based chunks for optimal processing
4. **Embedding**: Converts text chunks into vector embeddings using TF-IDF (with fallback to neural embeddings via Sentence Transformers)
5. **Vector Storage**: Stores embeddings in Pinecone vector database for efficient retrieval
6. **Question Answering**: Uses RAG (Retrieval-Augmented Generation) to answer questions by:
   - Finding relevant text chunks using vector similarity search
   - Generating answers using Google Gemini API with retrieved context

## Troubleshooting

### Common Issues

1. **API Key Errors**: Make sure your API keys are correctly set in the `.env` file
2. **Rate Limiting**: The system includes rate limiting, but you may need to wait if you hit Gemini's rate limits
3. **PDF Processing**: Ensure your PDF file is not corrupted and contains extractable text
4. **Memory Issues**: For large PDFs, consider processing smaller sections
5. **Embedding Issues**: If neural embeddings fail, the system automatically falls back to TF-IDF

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed correctly
2. Verify your API keys are valid and have sufficient credits
3. Ensure your PDF file is accessible and contains text
---

## Blog
Read about [Vector Database Architecture](https://arshad-kazi.com/vector-database-and-its-architecture/)
