FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "meteo_ml.serve:app", "--host", "0.0.0.0", "--port", "8000"]

