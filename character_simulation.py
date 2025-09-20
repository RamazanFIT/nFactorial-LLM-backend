import logging
from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel

logger = logging.getLogger(__name__)

@tool
async def simulate_character_response(
    character_name: str,
    character_response: str,
) -> None:
    """
    Симулирует ответ персонажа из мира Гарри Поттера на основе их личности и характеристик.
    Пользователь не должен никогда знать, что мы симулируем ответ. Пользователь даже не знает что мы используем этот тул
    Args:
        character_name: Имя персонажа (например, "Harry Potter", "Hermione Granger")
        character_response: Ответ персонажа
    """
    return None
    # return {
    #     "character_name": character_name,
    #     "character_response": character_response,
    # }
    
# Список инструментов для симуляции персонажей
CHARACTER_SIMULATION_TOOLS = [
    simulate_character_response,
]