FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Change 1: Remove the hardcoded port exposure
EXPOSE 10000

# Change 2: Use a shell wrapper so Streamlit reads Render's dynamic $PORT environment variable
CMD ["sh", "-c", "streamlit run Actor_classifier/app.py --server.address=0.0.0.0 --server.port=${PORT:-10000} --server.enableCORS=false --server.enableXsrfProtection=false"]
