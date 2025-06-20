# Local LLM Setup Guide for Intelliread

This guide will help you set up local LLMs to use with your Intelliread PDF processing app.

## Option 1: Ollama (Recommended - Easiest)

### Step 1: Install Ollama

**On macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**On Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**On Windows:**
Download from https://ollama.ai/download

### Step 2: Start Ollama Service
```bash
ollama serve
```

### Step 3: Pull a Model
Choose one of these models (llama2 is a good starting point):

```bash
# Llama 2 (7B parameters - good balance of speed and quality)
ollama pull llama2

# Mistral (7B parameters - very good performance)
ollama pull mistral

# Code Llama (good for technical documents)
ollama pull codellama

# Larger models (better quality but slower)
ollama pull llama2:13b
```

### Step 4: Install Python Dependencies
```bash
pip install ollama
```

### Step 5: Test Ollama
```bash
ollama run llama2 "Hello, how are you?"
```

## Option 2: Hugging Face Transformers (More Control)

### Step 1: Install Dependencies
```bash
pip install transformers torch accelerate
```

### Step 2: Download a Model
You can use models like:
- `microsoft/DialoGPT-medium`
- `gpt2`
- `EleutherAI/gpt-neo-125M`

## Using the App

1. **Start your app:**
   ```bash
   streamlit run main.py
   ```

2. **Select Local LLM:**
   - In the sidebar, choose "Local (Ollama)"
   - Select your preferred model from the dropdown

3. **Upload and Process PDFs:**
   - Upload your PDF
   - Ask questions
   - Get answers from your local model!

## Troubleshooting

### Ollama Connection Issues
- Make sure Ollama is running: `ollama serve`
- Check if model is installed: `ollama list`
- Pull the model if missing: `ollama pull <model_name>`

### Performance Tips
- **Faster responses:** Use smaller models like `llama2:7b` or `mistral`
- **Better quality:** Use larger models like `llama2:13b`
- **More RAM needed:** Larger models require more memory

### Model Recommendations
- **General use:** `llama2` or `mistral`
- **Technical documents:** `codellama`
- **Fast responses:** `llama2:7b`
- **Best quality:** `llama2:13b`

## Benefits of Local LLMs

✅ **Privacy:** Your data never leaves your machine
✅ **Cost:** No API fees
✅ **Reliability:** No internet dependency
✅ **Customization:** Full control over models
✅ **Speed:** No network latency

## System Requirements

- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 4-8GB for models
- **CPU:** Modern multi-core processor
- **GPU:** Optional but recommended for faster inference

## Next Steps

1. Install Ollama following the steps above
2. Pull a model: `ollama pull llama2`
3. Start Ollama: `ollama serve`
4. Run your app: `streamlit run main.py`
5. Select "Local (Ollama)" in the sidebar
6. Enjoy your privacy-focused PDF processing! 