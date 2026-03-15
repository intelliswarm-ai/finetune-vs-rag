#!/bin/bash
set -e

echo "============================================"
echo "  Fine-Tuning vs RAG Demo -- macOS Setup"
echo "============================================"

# -----------------------------------------------------------
# 0. Check prerequisites
# -----------------------------------------------------------
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 is required. Install via: brew install python"; exit 1; }
command -v ollama >/dev/null 2>&1 || { echo "ERROR: ollama is required. Install via: brew install ollama"; exit 1; }

# -----------------------------------------------------------
# 1. Ensure Ollama is running
# -----------------------------------------------------------
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

if ! curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve &
    OLLAMA_PID=$!
    echo "Waiting for Ollama to be ready..."
    RETRIES=0
    until curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; do
        RETRIES=$((RETRIES + 1))
        if [ "$RETRIES" -ge 30 ]; then
            echo "ERROR: Ollama did not start after 60 seconds."
            echo "Try starting it manually: ollama serve"
            exit 1
        fi
        sleep 2
    done
    echo "Ollama is running (PID: ${OLLAMA_PID})"
else
    echo "Ollama is already running at ${OLLAMA_HOST}"
fi

# -----------------------------------------------------------
# 2. Pull base model
# -----------------------------------------------------------
MODEL="${LLM_MODEL:-llama2}"
echo "Ensuring base model '${MODEL}' is available..."
ollama pull "${MODEL}" 2>/dev/null || echo "Model ${MODEL} already available"

# -----------------------------------------------------------
# 3. Create fine-tuned model (FinQA-7B = llama2 + LoRA)
# -----------------------------------------------------------
FT_MODEL="${FINETUNED_LLM_MODEL:-finqa-7b}"
if ! ollama list | grep -q "${FT_MODEL}"; then
    echo "Fine-tuned model '${FT_MODEL}' not found."
    echo "Downloading FinQA-7B LoRA adapter from HuggingFace (~128MB)..."

    ADAPTER_DIR="${HOME}/.finetune-vs-rag/finqa-adapter"
    mkdir -p "${ADAPTER_DIR}"

    python3 -c "
import os, shutil
from huggingface_hub import hf_hub_download

hf_model = 'truocpham/FinQA-7B-Instruct-v0.1'
adapter_dir = '${ADAPTER_DIR}'

adapter_files = ['adapter_config.json', 'adapter_model.safetensors',
                 'tokenizer.model', 'tokenizer.json', 'tokenizer_config.json',
                 'special_tokens_map.json', 'config.json']

for fname in adapter_files:
    try:
        path = hf_hub_download(
            repo_id=hf_model,
            filename=fname,
            token=os.environ.get('HF_TOKEN'),
        )
        dst = os.path.join(adapter_dir, fname)
        shutil.copy2(path, dst)
        size_mb = os.path.getsize(dst) / 1024 / 1024
        print(f'  Downloaded {fname} ({size_mb:.1f} MB)')
    except Exception as e:
        print(f'  Skipped {fname}: {e}')

print(f'Adapter files ready at {adapter_dir}')
" || echo "WARNING: LoRA adapter download failed."

    # Create Modelfile and import into Ollama
    MODELFILE="${ADAPTER_DIR}/Modelfile"
    cat > "${MODELFILE}" <<MEOF
FROM ${MODEL}
ADAPTER ${ADAPTER_DIR}
PARAMETER temperature 0.1
PARAMETER num_ctx 4096
MEOF

    echo "Creating Ollama model '${FT_MODEL}' (applying LoRA adapter)..."
    ollama create "${FT_MODEL}" -f "${MODELFILE}" || echo "WARNING: Fine-tuned model creation failed. Demo will run but fine-tuned queries may fail."
else
    echo "Fine-tuned model '${FT_MODEL}' already available."
fi

# -----------------------------------------------------------
# 4. Set up Python virtual environment
# -----------------------------------------------------------
VENV_DIR=".venv-macos"
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

echo "Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || pip install --quiet torch
pip install --quiet -r requirements-demo.txt

# -----------------------------------------------------------
# 5. Pre-download models (FinBERT, bert-base, embeddings)
# -----------------------------------------------------------
echo "Ensuring ML models are cached..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sentence_transformers import SentenceTransformer

print('  Checking FinBERT...')
AutoTokenizer.from_pretrained('ProsusAI/finbert')
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

print('  Checking bert-base-uncased...')
AutoTokenizer.from_pretrained('bert-base-uncased')
AutoModel.from_pretrained('bert-base-uncased')

print('  Checking embedding model...')
SentenceTransformer('all-MiniLM-L6-v2')

print('  All models ready.')
"

# -----------------------------------------------------------
# 6. Initialize RAG engine
# -----------------------------------------------------------
echo "Initializing RAG engine..."
python3 -c "
import sys; sys.path.insert(0, 'app')
from rag_engine import RAGEngine
engine = RAGEngine.get_instance()
engine.initialize()
print(f'RAG ready: {engine.num_chunks} chunks indexed')
" || echo "RAG init will happen on first query"

# -----------------------------------------------------------
# 7. Launch Streamlit
# -----------------------------------------------------------
echo ""
echo "============================================"
echo "  Starting demo at http://localhost:8501"
echo "============================================"
export OLLAMA_HOST="${OLLAMA_HOST}"
export LLM_MODEL="${MODEL}"
export FINETUNED_LLM_MODEL="${FT_MODEL}"

exec streamlit run app/finetune_vs_rag.py
