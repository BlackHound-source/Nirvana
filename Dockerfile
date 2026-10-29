
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "import os, zipfile; archive='/app/CMAPSSData.zip'; target='/app/data'; os.makedirs(target, exist_ok=True); z=zipfile.ZipFile(archive); z.extractall(target); z.close()"

ENV AERO_DATA_DIR=/app/data
ENV AERO_OUT_DIR=/app/runs
ENV PORT=10000
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /app/runs

EXPOSE 10000
CMD ["python", "server.py"]
