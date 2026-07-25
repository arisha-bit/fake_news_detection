FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model for claim extraction
RUN python -m spacy download en_core_web_sm

COPY . .

# Build FAISS evidence index from the dataset (runs once at image build time)
RUN python -m app.ml.embeddings.build_index

# Build CLIP reverse image index (runs once at image build time)
RUN python -m app.ml.clip.build_image_index

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]