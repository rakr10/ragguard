FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "starlette>=0.37" "uvicorn>=0.30"
COPY src ./src
COPY data ./data
ENV PYTHONPATH=/app/src \
    RAGGUARD_CORPUS_DIR=/app/data/corpus \
    RAGGUARD_BACKEND=extractive
RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "ragguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
