# Build stage
FROM python:3.11-slim AS build

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN python -m spacy download en_core_web_md

COPY . .

# Runtime stage
FROM python:3.11-slim

WORKDIR /app
COPY --from=build /usr/local /usr/local
COPY --from=build /app /app

EXPOSE 8080

CMD ["bash", "-c", "gunicorn -w 4 -k gthread --threads 8 -b 0.0.0.0:${PORT:-8080} wsgi:app"]
