FROM python:3.11-slim

WORKDIR /app

# System deps: curl for healthcheck, libgomp for faiss/torch CPU ops
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU-only FIRST from the official CPU index.
# This prevents pip from pulling CUDA wheels (500MB+ each) for a CPU container.
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    torchvision==0.18.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install all remaining dependencies (torch/torchvision already satisfied above)
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model for claim extraction + knowledge graph
RUN python -m spacy download en_core_web_sm

COPY . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

EXPOSE 8000

CMD ["./entrypoint.sh"]
