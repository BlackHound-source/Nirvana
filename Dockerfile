FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV AERO_DATA_DIR=/app/data AERO_OUT_DIR=/app/runs PORT=8000
EXPOSE 8000
CMD ["python", "server.py"]
