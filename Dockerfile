FROM python:3.11-slim

LABEL maintainer="IntelliSwarm AI"
LABEL description="Fine-Tuning vs RAG: Live Demo with FinBERT, Ollama, and real RAG"

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

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

# sentence-transformers embedding model (~90MB)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Embedding model downloaded')"

# Copy application code and data
COPY app/ app/
COPY src/ src/
COPY data/ data/

# Streamlit config
COPY streamlit-config.toml /root/.streamlit/config.toml

# Entrypoint handles Ollama readiness and RAG init
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
