FROM python:3.11-slim

LABEL maintainer="IntelliSwarm AI"
LABEL description="Fine-Tuning vs RAG: Live Demo with FinBERT, Ollama, and real RAG"

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Ollama CLI binary (needed for reliable LoRA adapter import)
# v0.5.13 uses .tgz; works on both amd64 (WSL) and arm64 (macOS Apple Silicon)
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then ARCH="arm64"; else ARCH="amd64"; fi && \
    curl -fsSL -L "https://github.com/ollama/ollama/releases/download/v0.5.13/ollama-linux-${ARCH}.tgz" \
        -o /tmp/ollama.tgz && \
    tar -xzf /tmp/ollama.tgz -C /usr && \
    rm /tmp/ollama.tgz

# Install PyTorch CPU-only (smaller than GPU version)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install demo dependencies
COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

# Pre-download models so the container starts fast
# FinBERT (~420MB fine-tuned sentiment model)
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('ProsusAI/finbert'); \
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert'); \
print('FinBERT downloaded')"

# bert-base-uncased (~440MB, same architecture as FinBERT but NOT fine-tuned)
RUN python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('bert-base-uncased'); \
AutoModel.from_pretrained('bert-base-uncased'); \
print('bert-base-uncased downloaded')"

# distilbert-base-uncased (~260MB, base architecture for spam detection benchmark)
RUN python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('distilbert-base-uncased'); \
AutoModel.from_pretrained('distilbert-base-uncased'); \
print('distilbert-base-uncased downloaded')"

# sentence-transformers embedding model (~90MB)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Embedding model downloaded')"

# Copy application code and data
COPY app/ app/
COPY src/ src/
COPY data/ data/

# Download fine-tuned DistilBERT spam detector checkpoint from Google Drive (~260MB)
RUN python -c "\
import sys; sys.path.insert(0, 'app'); \
from download_spam_model import download_checkpoint; \
download_checkpoint() \
" || echo "Spam model download will be attempted at runtime"

# Streamlit config
COPY streamlit-config.toml /root/.streamlit/config.toml

# Entrypoint handles Ollama readiness and RAG init
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
