FROM python:3.11-slim

WORKDIR /app

# ============================================================
# SYSTEM DEPENDENCIES
# curl     -> backend healthcheck (used in docker-compose)
# libgomp1 -> FAISS / PyTorch OpenMP CPU operations
# libglib2.0-0 -> required by opencv-python-headless
# ============================================================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# PYTHON DEPENDENCIES
# ============================================================
COPY requirements.txt .

# ============================================================
# PYTORCH CPU-ONLY
# Installed first so subsequent pip installs cannot pull a CUDA build.
#   torch       2.6.0+cpu
#   torchvision 0.21.0+cpu
# ============================================================
RUN pip install --no-cache-dir \
    torch==2.6.0+cpu \
    torchvision==0.21.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ============================================================
# ALL OTHER PYTHON DEPENDENCIES
# ============================================================
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# SPACY ENGLISH MODEL
# ============================================================
RUN python -m spacy download en_core_web_sm

# ============================================================
# APPLICATION CODE
# ============================================================
COPY . .

# ============================================================
# VERIFY DISTILBERT IMPORTS AT BUILD TIME
# This catches version mismatches before the container ever starts.
# ============================================================
RUN CUDA_VISIBLE_DEVICES=-1 python -c "\
import os; os.environ['CUDA_VISIBLE_DEVICES']='-1'; \
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification; \
print('DistilBERT imports OK')"

# ============================================================
# ENTRYPOINT
# ============================================================
RUN chmod +x entrypoint.sh

EXPOSE 8000

CMD ["./entrypoint.sh"]
