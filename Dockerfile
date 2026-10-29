
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Extract the dataset and search nested directories for required files.
RUN python -c "import os, zipfile; archive='/app/CMAPSSData.zip'; target='/app/data'; assert os.path.isfile(archive), 'CMAPSSData.zip missing from Docker build context'; os.makedirs(target, exist_ok=True); z=zipfile.ZipFile(archive); assert z.testzip() is None, 'Dataset ZIP is corrupt'; z.extractall(target); z.close(); required=['train_FD001.txt','test_FD001.txt','RUL_FD001.txt']; found={f for root,dirs,files in os.walk(target) for f in files}; missing=[f for f in required if f not in found]; print('Dataset directory:', target); print('Files found:', sorted(found)[:100]); assert not missing, 'Required dataset files missing: '+str(missing)"

ENV AERO_DATA_DIR=/app/data \
    AERO_OUT_DIR=/app/runs \
    PORT=10000 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /app/runs

EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 120 server:app"]
