FROM python:3.11-slim

WORKDIR /app

# Скопировать зависимости
COPY requirements.txt .

# Установить зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Скопировать код
COPY . .

# Запуск (например, для FastAPI)
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]
