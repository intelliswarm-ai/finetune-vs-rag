#!/bin/bash
set -e

echo "============================================"
echo "  Fine-Tuning vs RAG Demo -- Starting"
echo "============================================"

OLLAMA="${OLLAMA_HOST:-http://localhost:11434}"

# Wait for Ollama to be reachable
echo "Waiting for Ollama at ${OLLAMA}..."
RETRIES=0
MAX_RETRIES=30
until curl -sf "${OLLAMA}/api/tags" > /dev/null 2>&1; do
    RETRIES=$((RETRIES + 1))
    if [ "$RETRIES" -ge "$MAX_RETRIES" ]; then
        echo "WARNING: Ollama not reachable after ${MAX_RETRIES} attempts."
        echo "The demo will start but LLM queries will fail."
        echo "Make sure Ollama is running: ollama serve"
        break
    fi
    sleep 2
done

# -----------------------------------------------------------
# 1. Pull the BASE model (llama2) -- used for Base and RAG
# -----------------------------------------------------------
MODEL="${LLM_MODEL:-llama2}"
echo "Ensuring base model '${MODEL}' is available..."
if ! curl -sf "${OLLAMA}/api/tags" | grep -q "\"${MODEL}\"" 2>/dev/null; then
    echo "Pulling ${MODEL} (this may take a few minutes on first run)..."
    curl -sf "${OLLAMA}/api/pull" \
        -d "{\"name\": \"${MODEL}\"}" || echo "Model pull failed -- you may need to run: ollama pull ${MODEL}"
fi

# -----------------------------------------------------------
# 2. Create FINE-TUNED model (FinQA-7B) = llama2 + LoRA adapter
#    Downloads only the LoRA adapter (~128MB) from HuggingFace
#    Ollama applies it on top of llama2 (already downloaded)
# -----------------------------------------------------------
FT_MODEL="${FINETUNED_LLM_MODEL:-finqa-7b}"
echo "Ensuring fine-tuned model '${FT_MODEL}' is available..."
if ! curl -sf "${OLLAMA}/api/tags" | grep -q "\"${FT_MODEL}\"" 2>/dev/null; then
    echo "Fine-tuned model '${FT_MODEL}' not found in Ollama."
    echo "Downloading FinQA-7B LoRA adapter from HuggingFace (~128MB)..."

    python3 -c "
import os, json, shutil
from huggingface_hub import hf_hub_download

hf_model = 'truocpham/FinQA-7B-Instruct-v0.1'
adapter_dir = '/shared_models/finqa-adapter'
ollama_host = '${OLLAMA}'
ft_model = '${FT_MODEL}'
base_model = '${MODEL}'

os.makedirs(adapter_dir, exist_ok=True)

# Download ONLY the LoRA adapter files (not the full base model)
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

# Save Modelfile for CLI-based import (more reliable than API)
modelfile = f'''FROM {base_model}
ADAPTER /shared_models/finqa-adapter
PARAMETER temperature 0.1
PARAMETER num_ctx 4096
'''
with open('/shared_models/Modelfile', 'w') as f:
    f.write(modelfile)
print('Modelfile saved to /shared_models/Modelfile')
" || echo "FinQA-7B adapter download failed. Check logs for details."

    # Use Ollama CLI (more reliable than API for LoRA adapter conversion)
    echo "Creating Ollama model '${FT_MODEL}' via CLI (applying LoRA adapter)..."
    echo "This may take a few minutes..."
    OLLAMA_HOST="${OLLAMA}" ollama create "${FT_MODEL}" -f /shared_models/Modelfile \
        && echo "Fine-tuned model '${FT_MODEL}' created successfully!" \
        || echo "FinQA-7B import failed. Try manually: docker exec ollama ollama create ${FT_MODEL} -f /shared_models/Modelfile"
else
    echo "Fine-tuned model '${FT_MODEL}' already available."
fi

# Pre-initialize RAG (embed documents into ChromaDB)
echo "Initializing RAG engine (embedding documents)..."
python3 -c "
import sys; sys.path.insert(0, 'app')
from rag_engine import RAGEngine
engine = RAGEngine.get_instance()
engine.initialize()
print(f'RAG ready: {engine.num_chunks} chunks indexed')
" || echo "RAG init will happen on first query"

echo ""
echo "Starting Streamlit on port 8501..."
exec streamlit run app/finetune_vs_rag.py
