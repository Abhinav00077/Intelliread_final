import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import time
import torch
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# Add local LLM support
try:
    from langchain.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

def load_custom_css():
    """Load custom CSS for modern styling"""
    st.markdown("""
    <style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* File uploader styling */
    .upload-section {
        background: white;
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #764ba2;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
    }
    
    .upload-section h3 {
        color: #2c3e50;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .upload-section p {
        color: #34495e;
        margin: 0;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Response styling */
    .response-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .response-container h3 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-error {
        background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-info {
        background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .feature-card h3 {
        color: #2c3e50;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .feature-card p {
        color: #34495e;
        margin: 0;
        line-height: 1.5;
    }
    
    .feature-card ul {
        color: #34495e;
        margin: 0.5rem 0 0 0;
        padding-left: 1.5rem;
    }
    
    .feature-card li {
        margin-bottom: 0.25rem;
    }
    
    /* Loading animation */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 2rem 0 1rem 0;
        text-align: center;
        border-top: 2px solid #667eea;
    }
    
    .footer p {
        margin: 0;
        color: #6c757d;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .footer .author {
        color: #667eea;
        font-weight: 600;
        text-decoration: none;
    }
    
    .footer .author:hover {
        color: #764ba2;
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Set up page config with modern styling
    st.set_page_config(
        page_title="Intelliread - AI PDF Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Modern header with gradient background
    st.markdown("""
    <div class="main-header">
        <h1>📚 INTELLIREAD</h1>
        <p>Illuminating PDFs with Intelligent Answers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Smart Search</h3>
            <p>Advanced semantic search through your documents</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI Powered</h3>
            <p>Get intelligent answers from your PDF content</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🔒 Privacy First</h3>
            <p>Process documents locally or securely in the cloud</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model selection sidebar with modern styling
    st.sidebar.markdown("""
    <div class="sidebar-content">
        <h3>⚙️ Model Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    model_type = st.sidebar.selectbox(
        "🤖 Choose LLM Provider",
        ["Local (Ollama)", "OpenAI"],
        help="Select whether to use a local model via Ollama or OpenAI API"
    )
    
    if model_type == "Local (Ollama)":
        if not OLLAMA_AVAILABLE:
            st.sidebar.markdown("""
            <div class="status-error">
                ❌ Ollama not available
            </div>
            """, unsafe_allow_html=True)
            st.sidebar.info("Please install it with: `pip install ollama`")
            return
        
        # Ollama model selection
        ollama_model = st.sidebar.selectbox(
            "📋 Select Ollama Model",
            ["mistral", "tinyllama", "llama2", "codellama", "llama2:7b", "llama2:13b"],
            help="Choose which local model to use. Make sure you have it installed with 'ollama pull <model_name>'"
        )
        
        # Initialize Ollama LLM
        try:
            llm = Ollama(model=ollama_model)
            st.sidebar.markdown(f"""
            <div class="status-success">
                ✅ Connected to {ollama_model}
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.markdown(f"""
            <div class="status-error">
                ❌ Failed to connect to {ollama_model}
            </div>
            """, unsafe_allow_html=True)
            st.sidebar.info(f"Make sure Ollama is running and you have the model installed:")
            st.sidebar.code(f"ollama pull {ollama_model}")
            return
            
    else:  # OpenAI
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        llm = OpenAI()
        st.sidebar.markdown("""
        <div class="status-success">
            ✅ Using OpenAI API
        </div>
        """, unsafe_allow_html=True)
    
    # File upload section with modern styling
    st.markdown("""
    <div class="upload-section">
        <h3>📄 Upload Your PDF Document</h3>
        <p>Drag and drop your PDF file here or click to browse</p>
    </div>
    """, unsafe_allow_html=True)
    
    pdf = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
    
    # Text extraction with enhanced progress tracking
    if pdf is not None:
        st.success(f"✅ Successfully uploaded: {pdf.name}")
        
        with st.expander("📊 Processing Progress", expanded=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Extract text
            status_text.text("📖 Extracting text from PDF...")
            pdf_reader = PdfReader(pdf)
            text = ""
            for i, page in enumerate(pdf_reader.pages):
                text += page.extract_text()
                progress = int((i + 1) / len(pdf_reader.pages) * 25)
                progress_bar.progress(progress)
            
            # Split into chunks
            status_text.text("✂️ Splitting text into chunks...")
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=250,
                chunk_overlap=75,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            progress_bar.progress(50)
            
            # Convert chunks into embeddings
            status_text.text("🧠 Generating embeddings...")
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            embeddings = []
            for i, chunk in enumerate(chunks):
                input_ids = tokenizer.encode(chunk, return_tensors='pt')
                with torch.no_grad():
                    output = model(input_ids)
                    embedding = output.last_hidden_state[:,0,:].numpy()
                    embeddings.append(embedding.flatten().tolist())
                progress = 50 + int((i + 1) / len(chunks) * 25)
                progress_bar.progress(progress)
            
            # Pinecone setup
            status_text.text("☁️ Setting up vector database...")
            pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

            index_name = "testing"
            try:
                if index_name not in pc.list_indexes().names():
                    pc.create_index(
                        name=index_name,
                        dimension=384,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                        )
                    )
            except Exception as e:
                st.error(f"Could not create index: {e}")
                st.info("Please check your Pinecone account settings or upgrade your plan.")
                return
                
            indexer = pc.Index(index_name)
            
            # Prepare vectors
            vectors = []
            for i in range(len(embeddings)):
                vectors.append({'id': str(i), 'values': embeddings[i]})
            
            indexer.upsert(vectors)
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
        
        # Query section with modern styling
        st.markdown("""
        <div class="feature-card">
            <h3>❓ Ask Questions About Your Document</h3>
            <p>Type your question below and get intelligent answers based on your PDF content</p>
        </div>
        """, unsafe_allow_html=True)
        
        user_query = st.text_area(
            "💭 Enter your question here...",
            height=100,
            placeholder="e.g., What are the main points discussed in the document?"
        )
        
        if user_query:
            with st.spinner('🤔 Processing your query...'):
                # Generate query embeddings
                query_embeddings = []
                query_input_ids = tokenizer.encode(user_query, return_tensors='pt')
                with torch.no_grad():
                    output = model(query_input_ids)
                    query_embedding = output.last_hidden_state[:,0,:].numpy()
                    query_embeddings.append(query_embedding.flatten().tolist())
                
                # Search for relevant chunks
                search_results = indexer.query(vector=query_embeddings[0], top_k=5)
                ids = [result['id'] for result in search_results['matches']]
                
                selected_chunk = [chunks[int(ids[j])] for j in range(5)]
                
                # Generate response
                prompt = PromptTemplate(
                    input_variables=["query", "database"],
                    template="Answer this question: {query}\n\nUse knowledge from this text to generate an appropriate answer: {database}\n\nProvide a clear, comprehensive response:",
                )
                chain = LLMChain(llm=llm, prompt=prompt)
                
                if model_type == "OpenAI":
                    with get_openai_callback() as cb:
                        response = chain.run({
                            'query': user_query,
                            'database': selected_chunk
                        })
                        print(cb)
                else:
                    response = chain.run({
                        'query': user_query,
                        'database': selected_chunk
                    })
                
                # Display response with modern styling
                st.markdown("""
                <div class="response-container">
                    <h3>🤖 AI Response</h3>
                </div>
                """, unsafe_allow_html=True)
                st.write(response)
                
                # Show confidence scores
                with st.expander("📊 Search Confidence Scores"):
                    for i, result in enumerate(search_results['matches']):
                        st.write(f"Chunk {i+1}: {round(result['score'] * 100, 1)}% relevance")
    else:
        # Welcome message when no PDF is uploaded
        st.markdown("""
        <div class="feature-card">
            <h3>🚀 Get Started</h3>
            <p>Upload a PDF document above to begin asking questions and getting intelligent answers!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick tips
        st.markdown("""
        <div class="feature-card">
            <h3>💡 Tips for Better Results</h3>
            <ul>
                <li>Upload clear, text-based PDFs for best results</li>
                <li>Ask specific questions for more accurate answers</li>
                <li>Use local models for privacy-sensitive documents</li>
                <li>Try different models for various document types</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Professional footer with author credit
    st.markdown("""
    <div class="footer">
        <p>Engineered by <span class="author">Abhinav Pandey</span> | AI meets code</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
