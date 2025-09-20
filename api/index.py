# Vercel entry point
import sys
import os

# Добавляем корневую директорию в Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_main import app

# Vercel требует переменную app для развертывания
handler = app