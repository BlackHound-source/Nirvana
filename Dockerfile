
FROM python:3.11-slim

WORKDIR /app

# Install application dependencies, including Gunicorn
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, dashboard, and dataset archive
COPY . .

# Extract the NASA C-MAPSS dataset
# Fail the build if the ZIP is missing or invalid.
RUN python -c "import os, zipfile; archive='/app/CMAPSSData.zip'; target='/app/data'; os.makedirs(target, exist_ok=True); z=zipfile.ZipFile(archive); z.extractall(target); z.close(); required=['train_FD001.txt','test_FD001.txt','RUL_FD001.txt']; missing=[f for f in required if not os.path.isfile(os.path.join(target,f))]; print('Dataset files:', os.listdir(target)); assert not missing, 'Missing dataset files: '+', '.join(missing)"

# Configure application paths and runtime
ENV AERO_DATA_DIR=/app/data \
    AERO_OUT_DIR=/app/runs \
    PORT=10000 \
    PYTHONUNBUFFERED=1

# Create the output directory
RUN mkdir -p /app/runs

# Render web-service port
EXPOSE 10000

# Start Flask through Gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 120 server:app"]
