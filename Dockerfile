FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "meteo_ml.serve:app", "--host", "0.0.0.0", "--port", "8000"]

