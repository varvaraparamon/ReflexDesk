FROM python:3.12-slim

RUN pip install --no-cache-dir pandas numpy scipy

WORKDIR /opt/sandbox
COPY src/reflexdesk/__init__.py ./reflexdesk/__init__.py
COPY src/reflexdesk/sandbox ./reflexdesk/sandbox

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin sandbox \
    && mkdir -p /run/sandbox && chown sandbox /run/sandbox \
    && python -m compileall -q /opt/sandbox

ENV PYTHONPATH=/opt/sandbox \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

USER sandbox
CMD ["python", "-m", "reflexdesk.sandbox.server", "--socket", "/run/sandbox/sandbox.sock"]
