FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

RUN mkdir -p /app/llm_checkpoints

ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]
