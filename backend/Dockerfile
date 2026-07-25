FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model for claim extraction
RUN python -m spacy download en_core_web_sm

COPY . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

EXPOSE 8000

CMD ["./entrypoint.sh"]