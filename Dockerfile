FROM python:3.11-slim

WORKDIR /app

# Скопировать зависимости
COPY requirements.txt .

# Установить зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Скопировать код
COPY . .

# Запуск (например, для FastAPI)
CMD ["python", "main.py"]
