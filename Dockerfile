FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY artifacts/production_clone.joblib artifacts/production_clone.joblib
COPY artifacts/manifest.json artifacts/manifest.json

RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["sh", "-c", "vic-rent-ml serve --model artifacts/production_clone.joblib --host 0.0.0.0 --port ${PORT:-8000}"]
