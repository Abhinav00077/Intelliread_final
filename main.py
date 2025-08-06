#!/usr/bin/env python3
"""
Smart PDF Reader - Improved RAG System with Gemini API
An improved RAG system with better text search and fallback options
"""

import os
import nltk
import re
from dotenv import load_dotenv
from tqdm.auto import tqdm
import time
from collections import Counter
from pinecone import Pinecone
import numpy as np

# Local imports
from source.extract_text import extract_text_from_pdf, save_text_to_file
from source.cleaning_pipeline import TextFilter

# Load environment variables
load_dotenv()

class PineconeSearch:
    """Pinecone vector database integration for semantic search"""
    
    def __init__(self, api_key, index_name="pdfreader"):
        self.api_key = api_key
        self.index_name = index_name
        self.pc = None
        self.index = None
        self.initialized = False
        
    def initialize(self):
        """Initialize Pinecone and connect to existing serverless index"""
        try:
            # Initialize Pinecone with new client
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check if index exists
            indexes = self.pc.list_indexes()
            if self.index_name in [index.name for index in indexes]:
                self.index = self.pc.Index(self.index_name)
                self.initialized = True
                print(f"Successfully connected to existing serverless Pinecone index: {self.index_name}")
                # Minimal upsert test for debugging
                test_vector = [{
                    'id': 'test_id',
                    'values': [0.1] + [0.0] * 383,  # 384 floats with at least one non-zero value
                    'metadata': {'text': 'test', 'chunk_id': 0}
                }]
                try:
                    print("Testing upsert with a single vector...")
                    response = self.index.upsert(vectors=test_vector)
                    print("Test upsert successful!")
                except Exception as e:
                    print("Error during test upsert:", e)
                    import traceback
                    traceback.print_exc()
                return True
            else:
                print(f"Index '{self.index_name}' not found. Available indexes: {[index.name for index in indexes]}")
                print("For serverless Pinecone, make sure to create the index in the console first.")
                return False
            
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            return False
    
    def create_embeddings(self, chunks):
        """Create simple embeddings for text chunks using TF-IDF"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
            
            # Fit and transform the chunks
            tfidf_matrix = vectorizer.fit_transform(chunks)
            
            # Convert to dense array and pad/truncate to 384 dimensions
            embeddings = tfidf_matrix.toarray()
            
            # Pad or truncate to exactly 384 dimensions
            if embeddings.shape[1] < 384:
                # Pad with zeros
                padding = np.zeros((embeddings.shape[0], 384 - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, padding])
            elif embeddings.shape[1] > 384:
                # Truncate
                embeddings = embeddings[:, :384]
            
            print(f"Created {len(embeddings)} embeddings with {embeddings.shape[1]} dimensions")
            return embeddings
            
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            print("Falling back to text-based search...")
            return None
    
    def store_vectors(self, chunks, embeddings):
        """Store text chunks and their embeddings in Pinecone"""
        if not self.initialized:
            print("Pinecone not initialized. Cannot store vectors.")
            return False
        
        try:
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Ensure embedding is a list of Python floats
                values = [float(x) for x in embedding.tolist()]
                # Skip vectors that are all zeros
                if all(v == 0.0 for v in values):
                    print(f"Skipping chunk_{i} because its embedding is all zeros.")
                    continue
                vectors.append({
                    'id': f'chunk_{i}',
                    'values': values,
                    'metadata': {'text': str(chunk), 'chunk_id': int(i)}
                })
            # Debug: print the first vector to check format
            if vectors:
                print("Sample vector to upsert:", vectors[0])
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            print(f"Stored {len(vectors)} vectors in Pinecone")
            return True
            
        except Exception as e:
            print(f"Error storing vectors in Pinecone: {e}")
            return False
    
    def search(self, query, top_k=5):
        """Search for similar chunks using semantic similarity"""
        if not self.initialized:
            print("Pinecone not initialized. Cannot perform search.")
            return []
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Create TF-IDF vectorizer for query
            vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
            
            # Get all stored chunks from Pinecone
            stats = self.index.describe_index_stats()
            total_vectors = stats.total_vector_count
            
            if total_vectors == 0:
                print("No vectors found in Pinecone index")
                return []
            
            # Query Pinecone to get all vectors
            results = self.index.query(
                vector=[0.0] * 384,  # Dummy vector to get all
                top_k=total_vectors,
                include_metadata=True
            )
            
            # Extract chunks and create similarity matrix
            stored_chunks = [match.metadata['text'] for match in results.matches]
            
            # Create TF-IDF for stored chunks + query
            all_texts = stored_chunks + [query]
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarity between query and all stored chunks
            query_vector = tfidf_matrix[-1:]  # Last vector is the query
            chunk_vectors = tfidf_matrix[:-1]  # All except last are stored chunks
            
            similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
            
            # Get top-k most similar chunks
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            chunks = [stored_chunks[i] for i in top_indices if similarities[i] > 0]
            
            return chunks
            
        except Exception as e:
            print(f"Error searching in Pinecone: {e}")
            print("Falling back to text-based search...")
            return []

class GeminiAPI:
    """Wrapper for Gemini API calls"""
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.generate_url = f"{self.base_url}/models/gemini-1.5-flash:generateContent"
    
    def generate_text(self, prompt, max_tokens=1000):
        """Generate text using Gemini"""
        import requests
        import time
        
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': self.api_key
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.0
            }
        }
        
        # Try multiple times with different models
        models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro"
        ]
        
        for model in models:
            try:
                generate_url = f"{self.base_url}/models/{model}:generateContent"
                print(f"Trying model: {model}")
                
                response = requests.post(generate_url, headers=headers, json=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if 'candidates' in result and len(result['candidates']) > 0:
                        return result['candidates'][0]['content']['parts'][0]['text']
                    else:
                        print(f"No candidates in response for {model}")
                        continue
                elif response.status_code == 503:
                    print(f"Service unavailable for {model}, trying next model...")
                    time.sleep(1)
                    continue
                else:
                    print(f"Error {response.status_code} for {model}: {response.text}")
                    continue
                    
            except requests.exceptions.Timeout:
                print(f"Timeout for {model}, trying next model...")
                continue
            except Exception as e:
                print(f"Error with {model}: {e}")
                continue
        
        print("All models failed. Please check your API key and quota.")
        return None

class ImprovedTextSearch:
    """Improved text-based search using multiple strategies"""
    
    def __init__(self, chunks):
        self.chunks = chunks
        self.chunk_texts = [chunk.lower() for chunk in chunks]
        # Create word frequency index
        self.word_index = self._create_word_index()
    
    def _create_word_index(self):
        """Create a word frequency index for better search"""
        word_index = {}
        for i, chunk_text in enumerate(self.chunk_texts):
            words = re.findall(r'\w+', chunk_text)
            for word in words:
                if word not in word_index:
                    word_index[word] = []
                word_index[word].append(i)
        return word_index
    
    def search(self, query, k=5):
        """Search for relevant chunks using improved keyword matching"""
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        if not query_words:
            # If no words found, return first few chunks
            return self.chunks[:k]
        
        # Strategy 1: Direct word matching
        chunk_scores = Counter()
        for word in query_words:
            if word in self.word_index:
                for chunk_idx in self.word_index[word]:
                    chunk_scores[chunk_idx] += 1
        
        # Strategy 2: Partial word matching
        for word in query_words:
            for indexed_word, chunk_indices in self.word_index.items():
                if word in indexed_word or indexed_word in word:
                    for chunk_idx in chunk_indices:
                        chunk_scores[chunk_idx] += 0.5
        
        # Strategy 3: If no matches found, use semantic similarity
        if not chunk_scores:
            # Return chunks that contain any common words
            common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            for word in query_words:
                if word not in common_words:
                    for chunk_idx, chunk_text in enumerate(self.chunk_texts):
                        if word in chunk_text:
                            chunk_scores[chunk_idx] += 0.1
        
        # Get top k chunks
        top_chunks = []
        for chunk_idx, score in chunk_scores.most_common(k):
            if score > 0:
                top_chunks.append(self.chunks[chunk_idx])
        
        # If still no results, return first few chunks
        if not top_chunks:
            top_chunks = self.chunks[:k]
        
        return top_chunks

class SmartPDFReader:
    def __init__(self):
        """Initialize the Smart PDF Reader with API keys and configurations"""
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize Gemini API
        self.gemini = GeminiAPI(self.gemini_api_key)
        
        # Initialize Pinecone (optional)
        self.pinecone_search = None
        if self.pinecone_api_key:
            self.pinecone_search = PineconeSearch(self.pinecone_api_key)
            if self.pinecone_search.initialize():
                print("Pinecone integration enabled!")
            else:
                print("Pinecone initialization failed. Using text-based search only.")
                self.pinecone_search = None
        
        self.text_search = None
        self.chunks = []
    
    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt')
    
    def extract_and_clean_pdf(self, pdf_path, start_page=1, end_page=None, output_file="extracted_text.txt"):
        """Extract text from PDF and clean it"""
        print(f"Extracting text from {pdf_path}...")
        
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_path, start_page, end_page)
        if not extracted_text:
            raise ValueError("Failed to extract text from PDF")
        
        # Save extracted text
        save_text_to_file(extracted_text, output_file)
        print(f"Text extracted and saved to {output_file}")
        
        # Clean the text
        print("Cleaning extracted text...")
        text_filter = TextFilter(output_file)
        text_filter.clean_text()
        print("Text cleaning completed")
        
        return output_file
    
    def split_into_sentence_chunks(self, text, max_chunk_length=300):
        """Split text into sentence chunks"""
        from nltk.tokenize import sent_tokenize
        
        sentences = sent_tokenize(text)
        current_chunk = ""
        chunks = []
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk) >= 20]
        return chunks
    
    def ask_question(self, question):
        """Ask a question and get an answer using improved RAG"""
        if not self.text_search:
            print("No text search initialized. Please process a PDF first.")
            return None
        
        print(f"Question: {question}")
        print("Searching for relevant context...")
        
        # Try Pinecone search first if available
        relevant_chunks = []
        if self.pinecone_search and self.pinecone_search.initialized:
            print("Using Pinecone semantic search...")
            relevant_chunks = self.pinecone_search.search(question, top_k=5)
        
        # Fallback to text-based search if Pinecone fails or not available
        if not relevant_chunks:
            print("Using text-based search...")
            relevant_chunks = self.text_search.search(question, k=5)
        
        if not relevant_chunks:
            print("No relevant context found.")
            return None
        
        # Create context from relevant chunks
        context = "\n\n".join(relevant_chunks)
        
        # Create prompt for Gemini
        prompt = f"""Based on the following context from the document, please answer the question.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based only on the information provided in the context. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
        
        print("Generating answer using Gemini...")
        
        # Generate answer using Gemini
        answer = self.gemini.generate_text(prompt, max_tokens=1000)
        
        if answer:
            print(f"Answer: {answer}")
            return answer
        else:
            print("Failed to generate answer.")
            return None
    
    def process_pdf_and_setup(self, pdf_path, start_page=1, end_page=None):
        """Complete pipeline: extract, clean, and setup text search"""
        # Download NLTK data
        self.download_nltk_data()
        
        # Extract and clean PDF
        text_file = self.extract_and_clean_pdf(pdf_path, start_page, end_page)
        
        # Read cleaned text
        with open(text_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        # Split into chunks
        print("Splitting text into chunks...")
        self.chunks = self.split_into_sentence_chunks(text_content)
        print(f"Created {len(self.chunks)} text chunks")
        
        # Setup text-based search (always available as fallback)
        self.text_search = ImprovedTextSearch(self.chunks)
        
        # Setup Pinecone if available
        if self.pinecone_search and self.pinecone_search.initialized:
            print("Creating embeddings and storing in Pinecone...")
            embeddings = self.pinecone_search.create_embeddings(self.chunks)
            if embeddings is not None:
                self.pinecone_search.store_vectors(self.chunks, embeddings)
                print("Pinecone setup completed!")
            else:
                print("Failed to create embeddings. Using text-based search only.")
        
        print("PDF processing and setup completed successfully!")
        return True

def main():
    """Main function to run the Smart PDF Reader"""
    print("=== Smart PDF Reader - Improved RAG System with Gemini API ===")
    
    # Check if we have a PDF file to process
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the current directory.")
        print("Please place a PDF file in the project directory and run again.")
        return
    
    # Use the first PDF file found
    pdf_path = pdf_files[0]
    print(f"Found PDF file: {pdf_path}")
    
    try:
        # Initialize the reader
        reader = SmartPDFReader()
        
        # Process the PDF and setup the system
        reader.process_pdf_and_setup(pdf_path)
        
        # Interactive question-answering
        print("\n=== Interactive Q&A Session ===")
        print("You can now ask questions about the document content.")
        print("Example questions:")
        print("- What are the main topics discussed?")
        print("- How does the author explain [specific concept]?")
        print("- What are the key findings or conclusions?")
        print("- How does this relate to [specific topic]?")
        print("Type 'quit' to exit.")
        
        while True:
            question = input("\nEnter your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if question:
                try:
                    reader.ask_question(question)
                except Exception as e:
                    print(f"Error: {e}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have set up your environment variables:")
        print("1. GEMINI_API_KEY - Get from https://makersuite.google.com/app/apikey")
        print("2. PINECONE_API_KEY - Get from https://app.pinecone.io/")
        print("3. PINECONE_ENVIRONMENT - Your Pinecone environment")

if __name__ == "__main__":
    main() 