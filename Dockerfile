# Imagen autocontenida: instala dependencias, entrena durante el build y sirve
# la interfaz web y la API desde el mismo puerto.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY train_model.py api_modelo.py index.html app.js clientes_churn.csv ./
RUN python train_model.py --csv clientes_churn.csv --output artifacts

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api_modelo:app --host 0.0.0.0 --port ${PORT}"]
