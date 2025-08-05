#!/usr/bin/env python3
"""
Setup script for Smart PDF Reader
"""

import os
import subprocess
import sys

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"Python version: {sys.version}")

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = ".env"
    if not os.path.exists(env_file):
        print("Creating .env file...")
        with open(env_file, "w") as f:
            f.write("# Gemini API Key - Get from https://makersuite.google.com/app/apikey\n")
            f.write("GEMINI_API_KEY=your_gemini_api_key_here\n\n")
            f.write("# Pinecone API Key - Get from https://app.pinecone.io/\n")
            f.write("PINECONE_API_KEY=your_pinecone_api_key_here\n\n")
            f.write("# Pinecone Environment - Your Pinecone environment (e.g., 'us-east1-gcp')\n")
            f.write("PINECONE_ENVIRONMENT=your_pinecone_environment_here\n")
        print(".env file created. Please edit it with your API keys.")
    else:
        print(".env file already exists.")

def main():
    """Main setup function"""
    print("=== Smart PDF Reader Setup ===")
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Create .env file
    create_env_file()
    
    print("\n=== Setup Complete ===")
    print("Next steps:")
    print("1. Edit the .env file with your API keys:")
    print("   - Get Gemini API key from: https://makersuite.google.com/app/apikey")
    print("   - Get Pinecone API key from: https://app.pinecone.io/")
    print("2. Place a PDF file in the project directory")
    print("3. Run: python main.py")

if __name__ == "__main__":
    main() 