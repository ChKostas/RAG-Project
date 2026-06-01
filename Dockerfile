FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY preprocessed_db ./preprocessed_db_seed
COPY Knowledge-Base-Incurance ./Knowledge-Base-Incurance

EXPOSE 7860

CMD ["python", "src/app.py"]