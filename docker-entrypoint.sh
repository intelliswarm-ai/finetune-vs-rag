#!/bin/bash
set -e

echo "============================================"
echo "  Fine-Tuning vs RAG Demo -- Starting"
echo "============================================"

# Wait for Ollama to be reachable
echo "Waiting for Ollama at ${OLLAMA_HOST:-http://localhost:11434}..."
RETRIES=0
MAX_RETRIES=30
until curl -sf "${OLLAMA_HOST:-http://localhost:11434}/api/tags" > /dev/null 2>&1; do
    RETRIES=$((RETRIES + 1))
    if [ "$RETRIES" -ge "$MAX_RETRIES" ]; then
        echo "WARNING: Ollama not reachable after ${MAX_RETRIES} attempts."
        echo "The demo will start but LLM queries will fail."
        echo "Make sure Ollama is running: ollama serve"
        break
    fi
    sleep 2
done

# Pull the model if not already available
MODEL="${LLM_MODEL:-mistral}"
echo "Ensuring model '${MODEL}' is available..."
if ! curl -sf "${OLLAMA_HOST:-http://localhost:11434}/api/tags" | grep -q "\"${MODEL}\"" 2>/dev/null; then
    echo "Pulling ${MODEL} (this may take a few minutes on first run)..."
    curl -sf "${OLLAMA_HOST:-http://localhost:11434}/api/pull" \
        -d "{\"name\": \"${MODEL}\"}" || echo "Model pull failed -- you may need to run: ollama pull ${MODEL}"
fi

# Pre-initialize RAG (embed documents into ChromaDB)
echo "Initializing RAG engine (embedding documents)..."
python -c "
import sys; sys.path.insert(0, 'app')
from rag_engine import RAGEngine
engine = RAGEngine.get_instance()
engine.initialize()
print(f'RAG ready: {engine.num_chunks} chunks indexed')
" || echo "RAG init will happen on first query"

echo ""
echo "Starting Streamlit on port 8501..."
exec streamlit run app/app.py
