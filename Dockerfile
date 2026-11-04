
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-prod.txt .

RUN pip install --no-cache-dir -r requirements-prod.txt

COPY src/ ./src/
COPY models/sentiment_pipeline.joblib ./models/sentiment_pipeline.joblib
COPY app_api.py .

EXPOSE 7860

CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "7860"]
