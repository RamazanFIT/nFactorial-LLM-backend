from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import json
import logging
import os
import uuid
import httpx
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

# Простая версия без LangChain для Vercel
class SearchRequest(BaseModel):
    search: str
    thread_id: Optional[str] = None

class SimpleHarryPotterBot:
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.base_url = "https://api.anthropic.com/v1/messages"
        
    async def get_character_info(self, character_name: str):
        """Получение информации о персонаже через HP API"""
        try:
            async with httpx.AsyncClient() as client:
                # Поиск всех персонажей
                response = await client.get("https://hp-api.onrender.com/api/characters")
                characters = response.json()
                
                # Поиск по имени
                found = [c for c in characters if character_name.lower() in c.get('name', '').lower()]
                return found[:3] if found else None
        except Exception as e:
            logger.error(f"Error fetching character: {e}")
            return None
    
    async def stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Потоковый ответ от Claude"""
        try:
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            # Добавляем контекст о Harry Potter
            system_prompt = """Ты - эксперт по миру Гарри Поттера. Отвечай с энтузиазмом и знанием дела о персонажах, заклинаниях, событиях книг. Используй магические эмодзи ✨🪄⚡"""
            
            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user", 
                        "content": f"{system_prompt}\n\nВопрос: {prompt}"
                    }
                ],
                "stream": True
            }
            
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", 
                    self.base_url, 
                    headers=headers, 
                    json=payload,
                    timeout=30.0
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                                
                            try:
                                data = json.loads(data_str)
                                if data.get("type") == "content_block_delta":
                                    text = data.get("delta", {}).get("text", "")
                                    if text:
                                        yield f"data: {json.dumps({'content': text}, ensure_ascii=False)}\n\n"
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': 'Произошла магическая ошибка'}, ensure_ascii=False)}\n\n"

# Инициализация бота
bot = SimpleHarryPotterBot()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Simple Harry Potter API initialized")
    yield
    logger.info("Shutting down")

app = FastAPI(
    title="Harry Potter Simple API",
    description="Простой API для мира Гарри Поттера (Vercel)",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def generate_magical_response(
    search: str,
    thread_id: Optional[str]
) -> AsyncGenerator[str, None]:
    try:
        # Генерируем новый thread_id если не передан
        if not thread_id or thread_id == "null":
            thread_id = str(uuid.uuid4())
        
        # Проверяем, запрашивается ли информация о персонаже
        if any(keyword in search.lower() for keyword in ["персонаж", "character", "герой", "информация о"]):
            # Пытаемся извлечь имя персонажа
            for name in ["harry potter", "гарри поттер", "hermione", "гермиона", "ron", "рон", "snape", "снейп", "dumbledore", "дамблдор"]:
                if name in search.lower():
                    char_info = await bot.get_character_info(name)
                    if char_info:
                        search += f"\n\nДополнительная информация: {json.dumps(char_info[0], ensure_ascii=False)}"
                    break
        
        async for chunk in bot.stream_response(search):
            yield chunk
            
        yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        yield f"data: {json.dumps({'error': 'Произошла магическая ошибка'}, ensure_ascii=False)}\n\n"

@app.post("/chat/stream")
@limiter.limit("30/minute")
async def magical_chat_stream(
    search_request: SearchRequest,
    request: Request
):
    if not search_request.search.strip():
        raise HTTPException(status_code=400, detail="Поиск не может быть пустым")
    
    logger.info(f"Magical search: {search_request.search[:50]}...")
    
    return StreamingResponse(
        generate_magical_response(search_request.search, search_request.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/character/{character_name}")
async def get_character_card(character_name: str):
    """Получить информацию о персонаже"""
    try:
        char_info = await bot.get_character_info(character_name)
        if char_info:
            return {
                "character": character_name,
                "info": char_info,
                "type": "character_card"
            }
        else:
            raise HTTPException(status_code=404, detail="Персонаж не найден")
    except Exception as e:
        logger.error(f"Error getting character: {e}")
        raise HTTPException(status_code=500, detail="Ошибка получения персонажа")

@app.get("/spells")
async def get_spells():
    """Получить список заклинаний"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://hp-api.onrender.com/api/spells")
            return response.json()
    except Exception as e:
        logger.error(f"Error getting spells: {e}")
        raise HTTPException(status_code=500, detail="Ошибка получения заклинаний")

@app.get("/health")
async def health_check():
    return {"status": "Магия работает!", "service": "Harry Potter Simple API"}

@app.get("/")
async def root():
    return {
        "message": "🪄 Простой магический API мира Гарри Поттера!",
        "endpoints": {
            "stream_chat": "/chat/stream",
            "character_info": "/character/{character_name}",
            "spells": "/spells",
            "health": "/health"
        },
        "note": "Optimized for Vercel deployment"
    }

# Экспорт для Vercel готов