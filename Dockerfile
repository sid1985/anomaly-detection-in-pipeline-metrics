FROM python:3.11-slim

WORKDIR /app

# Install slim API-only deps (no jupyter/matplotlib/scipy)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Application source and trained model artefacts
COPY src/              ./src/
COPY azure/api.py      ./azure/api.py
COPY azure/__init__.py ./azure/__init__.py
# results/ must contain trained models before docker build runs (done in CI)
COPY results/          ./results/

EXPOSE 8000

# Disable TF verbose output at runtime
ENV TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONUNBUFFERED=1

CMD ["uvicorn", "azure.api:app", "--host", "0.0.0.0", "--port", "8000"]
