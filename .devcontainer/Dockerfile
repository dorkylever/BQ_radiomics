
FROM python:3.9

# Install R from Debian's own repos (Debian 12 "bookworm" ships R 4.2.x).
RUN apt-get update \
    && apt-get install -y --no-install-recommends r-base \
    && rm -rf /var/lib/apt/lists/*

# Install bq-radiomics (and all its dependencies) straight from PyPI.
RUN pip install bq-radiomics \
    && rm -rf /root/.cache/pip