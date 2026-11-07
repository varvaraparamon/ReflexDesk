FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends pandoc fonts-dejavu-core libpango-1.0-0 libpangoft2-1.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/reflexdesk
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . weasyprint

COPY prompts ./prompts
COPY skills ./skills
COPY scripts ./scripts

ENV REFLEXDESK_HOME=/opt/reflexdesk \
    REFLEXDESK_DATA_DIR=/data \
    PYTHONUNBUFFERED=1

RUN useradd --uid 10000 --create-home desk && mkdir -p /data /state && chown desk /data /state
USER desk

EXPOSE 8000
CMD ["reflexdesk", "serve", "--host", "0.0.0.0", "--port", "8000"]
