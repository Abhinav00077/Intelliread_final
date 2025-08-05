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

# Local imports
from source.extract_text import extract_text_from_pdf, save_text_to_file
from source.cleaning_pipeline import TextFilter

# Load environment variables
load_dotenv()

class GeminiAPI:
    """Wrapper for Gemini API calls"""
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.generate_url = f"{self.base_url}/models/gemini-2.0-flash:generateContent"
    
    def generate_text(self, prompt, max_tokens=1000):
        """Generate text using Gemini"""
        import requests
        
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
        
        try:
            response = requests.post(self.generate_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            print(f"Error generating text: {e}")
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
        
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize Gemini API
        self.gemini = GeminiAPI(self.gemini_api_key)
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
        
        # Search for relevant chunks
        relevant_chunks = self.text_search.search(question, k=5)
        
        if not relevant_chunks:
            print("No relevant context found.")
            return None
        
        # Create context from relevant chunks
        context = "\n\n".join(relevant_chunks)
        
        # Create prompt for Gemini
        prompt = f"""Based on the following context from Marcus Aurelius' Meditations, please answer the question.

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
        
        # Setup improved text search
        self.text_search = ImprovedTextSearch(self.chunks)
        
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
        print("You can now ask questions about Marcus Aurelius' Meditations.")
        print("Example questions:")
        print("- What does Marcus Aurelius say about death?")
        print("- How should one deal with difficult people?")
        print("- What are his thoughts on virtue?")
        print("- How does he view external circumstances?")
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

if __name__ == "__main__":
    main() 