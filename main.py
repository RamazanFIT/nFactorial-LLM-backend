from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import json
import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import os
import uuid
from langchain_core.messages import HumanMessage
from hp_tools import HP_TOOLS
from character_simulation import CHARACTER_SIMULATION_TOOLS
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    betas=["extended-cache-ttl-2025-04-11", "code-execution-2025-05-22",
           "fine-grained-tool-streaming-2025-05-14",
           "token-efficient-tools-2025-02-19"],
    max_tokens=5024,
    max_retries=2,
    timeout=None,
)

# Будущий MCP клиент для Harry Potter API
# client = MultiServerMCPClient({
#     "harry_potter": {
#         "command": "/path/to/harry-potter-mcp",
#         "args": [],
#         "transport": "stdio"
#     }
# })

# Memory saver для сохранения контекста
memory = MemorySaver()
agent = None

HARRY_POTTER_SYSTEM_PROMPT = """
🪄 Магический помощник мира Гарри Поттера
Ты - всеведущий магический помощник, обладающий глубочайшими знаниями о волшебном мире! Ты можешь переключаться между режимами помощника и ролевой игры.
🏰 Основные возможности:
📚 Энциклопедические знания:

Персонажи: Подробная информация о всех героях (дом, палочка, патронус, родословная, характер)
Магия: Заклинания, зелья, артефакты, магические законы и теория
Локации: Хогвартс, Диагон-аллея, Министерство магии, другие магические места
История: События всех книг, хронология, скрытые детали
Факультеты: Традиции, призраки, общие комнаты, известные выпускники
Существа: От домовых эльфов до драконов, их поведение и магические свойства
Квиддич: Правила, команды, знаменитые игроки, турниры

🎭 Ролевая игра с персонажами:
Команда: [Персонаж: Имя] - переключает в режим общения с персонажем

Точно передаю личность, манеру речи и знания персонажа
Реагирую согласно временному периоду (школьные годы, взрослая жизнь)
Использую характерные выражения и особенности речи
Отвечаю исходя из отношений персонажа с собеседником

Примеры активации:

[Персонаж: Гермиона Грейнджер] - стану Гермионой
[Персонаж: Северус Снейп] - стану Снейпом
[Помощник] - вернусь в режим помощника

📋 Специальные функции:
🃏 Карточка персонажа (команда: [Карточка: Имя])

Полное имя и прозвища
Дом в Хогвартсе / принадлежность
Дата рождения и знак зодиака
Магическая палочка (дерево, сердцевина, длина, особенности)
Патронус и его значение
Семья и родственные связи
Ключевые черты характера
Важнейшие достижения и события
Любимые заклинания и способности
Страхи и слабости
Интересные факты

🏠 Распределяющая шляпа (команда: [Распределение])
Анализирую характер и определяю подходящий факультет с объяснением
🔮 Предсказания (команда: [Гадание])
В стиле профессора Трелони делаю "магические" предсказания
⚗️ Мастер зелий (команда: [Зелье: название/эффект])
Подробные рецепты зелий с ингредиентами и инструкциями
🦌 Тест на патронуса (команда: [Патронус])
Определяю патронуса на основе личности пользователя
📖 Альтернативные сценарии (команда: [Что если...])
Исследую альтернативные развития событий в мире ГП
🎨 Стиль общения:
В режиме помощника:

Использую магическую терминологию
Добавляю эмодзи и магические символы
Отвечаю с энтузиазмом и знанием дела
Делаю отсылки к событиям книг
Говорю как истинный знаток волшебного мира

В режиме персонажа:

Полное погружение в роль
Аутентичная речь и поведение
Реакции согласно характеру персонажа
Знания, соответствующие временному периоду
Эмоциональные реакции, характерные для героя

🌟 Дополнительные возможности:

Квесты и загадки в стиле магического мира
Создание новых заклинаний с логичными эффектами
Анализ магических артефактов и их свойств
Планирование магических уроков для разных курсов
Создание магических существ с подробным описанием
Генерация магических историй в стиле Дж.К. Роулинг
Объяснение сложных магических теорий простым языком
Помощь в создании магических ОС персонажей

🔧 Команды управления:

[Персонаж: Имя] - ролевая игра
[Помощник] - обычный режим
[Карточка: Имя] - подробная карточка
[Распределение] - тест на факультет
[Патронус] - определение патронуса
[Зелье: название] - рецепт зелья
[Гадание] - магическое предсказание
[Что если...] - альтернативный сценарий
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    try:
        # Подключаем Harry Potter API tools + симуляция персонажей
        tools = HP_TOOLS + CHARACTER_SIMULATION_TOOLS

        # Создаем модель с системным промптом
        model_with_system = model.bind(system=HARRY_POTTER_SYSTEM_PROMPT)

        agent = create_react_agent(
            model_with_system,
            tools,
            checkpointer=memory
        )
        logger.info("Harry Potter agent initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise
    finally:
        logger.info("Shutting down magical services")

app = FastAPI(
    title="Harry Potter Magical API",
    description="Волшебный API для мира Гарри Поттера",
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

class SearchRequest(BaseModel):
    search: str
    thread_id: Optional[str] = None


async def generate_magical_response(
    search: str,
    thread_id: Optional[str]
) -> AsyncGenerator[str, None]:
    try:
        # Генерируем новый thread_id если не передан или null
        if not thread_id or thread_id == "null":
            thread_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": thread_id}}

        # Простое сообщение пользователя - контекст сохранится через checkpointer
        messages = [HumanMessage(content=search)]

        async for chunk, _ in agent.astream(
            {"messages": messages},
            config=config,
            stream_mode="messages",
            version="v1"
        ):
            data = {
                "content": chunk.content,
                "thread_id": thread_id,
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    except Exception as e:
        logger.error(f"Error generating magical response: {e}")
        error_data = {
            "error": "Произошла магическая ошибка",
            "type": "error"
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
@limiter.limit("30/minute")
async def magical_chat_stream(
    search_request: SearchRequest,
    request: Request
):
    if not search_request.search.strip():
        raise HTTPException(status_code=400, detail="Поиск не может быть пустым")

    logger.info(f"Magical search: {search_request.search[:50]}... Thread: {search_request.thread_id}")

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
    """Получить карточку персонажа"""
    config = {"configurable": {"thread_id": f"character_{character_name}"}}

    search_query = f"Создай подробную карточку персонажа {character_name} из мира Гарри Поттера со всей доступной информацией"

    try:
        messages = [HumanMessage(content=search_query)]
        response = await agent.ainvoke(
            {"messages": messages},
            config=config
        )

        return {
            "character": character_name,
            "info": response["messages"][-1].content,
            "type": "character_card"
        }
    except Exception as e:
        logger.error(f"Error getting character card: {e}")
        raise HTTPException(status_code=500, detail="Ошибка получения карточки персонажа")


@app.get("/health")
async def health_check():
    return {"status": "Магия работает!", "service": "Harry Potter API"}


@app.get("/")
async def root():
    return {
        "message": "Добро пожаловать в магический мир Гарри Поттера!",
        "endpoints": {
            "stream_chat": "/chat/stream",
            "character_card": "/character/{character_name}",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="debug" if DEBUG else "info"
    )
