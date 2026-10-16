FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py eval.py docker-entrypoint.py ./
COPY solver/ ./solver/
COPY configs/ ./configs/
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# Run as root so grader-mounted input files with mode 600 remain readable.
RUN mkdir -p /output /data
ENTRYPOINT ["python", "/app/docker-entrypoint.py"]
