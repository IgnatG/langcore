FROM python:3.14-slim-bookworm

WORKDIR /app

RUN pip install langcore

COPY demo_ollama.py .

CMD ["python", "demo_ollama.py"]
