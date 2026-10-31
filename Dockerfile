FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Copy and install
COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir -e .

COPY . .

# Generate demo data
RUN python demo/train.py || true

EXPOSE 8000

CMD ["uvicorn", "modelgate.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
