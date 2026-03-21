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

# -----------------------------------------------------------
# 3. Download fine-tuned DistilBERT spam checkpoint if missing
# -----------------------------------------------------------
echo "Ensuring fine-tuned spam detector checkpoint is available..."
if [ ! -f "models/spam_detector/checkpoint.pt" ]; then
    echo "Spam model checkpoint not found, downloading..."
    python3 -c "
import sys; sys.path.insert(0, 'app')
from download_spam_model import download_checkpoint
download_checkpoint()
" || echo "Spam model download failed. Run: python app/download_spam_model.py"
else
    echo "Spam model checkpoint already available."
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

# -----------------------------------------------------------
# 5. Pre-compute benchmark results so pages load instantly
# -----------------------------------------------------------

# Standard benchmark (50 test cases across 4 experiments)
if [ ! -f "data/benchmark_results.json" ]; then
    echo "Pre-computing standard benchmark results (50 test cases)..."
    python3 app/benchmark.py \
        && echo "Standard benchmark results saved to data/benchmark_results.json" \
        || echo "Standard benchmark pre-computation failed (will run on demand)"
else
    echo "Standard benchmark results already pre-computed."
fi

# Adversarial stress test (120 adversarial cases across 4 experiments)
JUDGE_FLAG=""
if [ -n "${OPENAI_API_KEY}" ]; then
    JUDGE_FLAG="--with-judge"
    echo "OPENAI_API_KEY detected -- will run adversarial benchmark with LLM-as-Judge (${JUDGE_MODEL:-gpt-4o})"
fi

if [ ! -f "data/adversarial_results.json" ]; then
    echo "Pre-computing adversarial stress test results (120 test cases)..."
    python3 app/adversarial_benchmark.py ${JUDGE_FLAG} \
        && echo "Adversarial results saved to data/adversarial_results.json" \
        || echo "Adversarial benchmark pre-computation failed (will run on demand)"
else
    echo "Adversarial stress test results already pre-computed."
fi

# RAG strengths benchmark (30 cases showing RAG advantages)
if [ ! -f "data/rag_strengths_results.json" ]; then
    echo "Pre-computing RAG strengths benchmark results (30 test cases)..."
    python3 app/rag_strengths_benchmark.py ${JUDGE_FLAG} \
        && echo "RAG strengths results saved to data/rag_strengths_results.json" \
        || echo "RAG strengths benchmark pre-computation failed (will run on demand)"
else
    echo "RAG strengths benchmark results already pre-computed."
fi

echo ""
echo "Starting Streamlit on port 8501..."
exec streamlit run app/Finetune_vs_RAG.py
