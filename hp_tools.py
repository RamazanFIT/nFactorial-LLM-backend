import httpx
import logging
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

HP_API_BASE_URL = "https://hp-api.onrender.com/api"

class Character(BaseModel):
    id: str
    name: str
    alternate_names: List[str]
    species: str
    gender: str
    house: str
    dateOfBirth: Optional[str]
    yearOfBirth: Optional[int]
    wizard: bool
    ancestry: str
    eyeColour: str
    hairColour: str
    wand: Dict[str, Any]
    patronus: str
    hogwartsStudent: bool
    hogwartsStaff: bool
    actor: str
    alternate_actors: List[str]
    alive: bool
    image: str

class Spell(BaseModel):
    id: str
    name: str
    description: str

@tool
async def get_all_characters() -> List[Dict[str, Any]]:
    """
    Получить информацию обо всех персонажах мира Гарри Поттера.
    Возвращает полный список персонажей с их характеристиками.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HP_API_BASE_URL}/characters")
            response.raise_for_status()
            characters = response.json()
            logger.info(f"Получено {len(characters)} персонажей")
            return characters
    except Exception as e:
        logger.error(f"Ошибка получения персонажей: {e}")
        return {"error": f"Не удалось получить персонажей: {str(e)}"}

@tool
async def get_character_by_id(character_id: str) -> Dict[str, Any]:
    """
    Получить подробную информацию о персонаже по его ID.
    
    Args:
        character_id: Уникальный идентификатор персонажа
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HP_API_BASE_URL}/character/{character_id}")
            response.raise_for_status()
            character_data = response.json()
            if character_data:
                logger.info(f"Получен персонаж: {character_data[0].get('name', 'Unknown')}")
                return character_data[0]
            return {"error": "Персонаж не найден"}
    except Exception as e:
        logger.error(f"Ошибка получения персонажа {character_id}: {e}")
        return {"error": f"Не удалось получить персонажа: {str(e)}"}

@tool
async def get_hogwarts_students() -> List[Dict[str, Any]]:
    """
    Получить список всех учеников Хогвартса.
    Возвращает информацию о студентах, которые учились в школе чародейства и волшебства.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HP_API_BASE_URL}/characters/students")
            response.raise_for_status()
            students = response.json()
            logger.info(f"Получено {len(students)} учеников Хогвартса")
            return students
    except Exception as e:
        logger.error(f"Ошибка получения учеников: {e}")
        return {"error": f"Не удалось получить учеников: {str(e)}"}

@tool
async def get_hogwarts_staff() -> List[Dict[str, Any]]:
    """
    Получить список всех сотрудников Хогвартса.
    Возвращает информацию о преподавателях и персонале школы.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HP_API_BASE_URL}/characters/staff")
            response.raise_for_status()
            staff = response.json()
            logger.info(f"Получено {len(staff)} сотрудников Хогвартса")
            return staff
    except Exception as e:
        logger.error(f"Ошибка получения сотрудников: {e}")
        return {"error": f"Не удалось получить сотрудников: {str(e)}"}

@tool
async def get_characters_by_house(house: str) -> List[Dict[str, Any]]:
    """
    Получить список персонажей определенного факультета Хогвартса.
    
    Args:
        house: Название факультета (gryffindor, slytherin, ravenclaw, hufflepuff)
    """
    valid_houses = ["gryffindor", "slytherin", "ravenclaw", "hufflepuff"]
    house_lower = house.lower()
    
    if house_lower not in valid_houses:
        return {"error": f"Неверный факультет. Допустимые значения: {', '.join(valid_houses)}"}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HP_API_BASE_URL}/characters/house/{house_lower}")
            response.raise_for_status()
            house_members = response.json()
            logger.info(f"Получено {len(house_members)} персонажей из {house.capitalize()}")
            return house_members
    except Exception as e:
        logger.error(f"Ошибка получения персонажей факультета {house}: {e}")
        return {"error": f"Не удалось получить персонажей факультета: {str(e)}"}

@tool
async def get_all_spells() -> List[Dict[str, Any]]:
    """
    Получить список всех заклинаний из мира Гарри Поттера.
    Возвращает заклинания с их названиями и описаниями.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HP_API_BASE_URL}/spells")
            response.raise_for_status()
            spells = response.json()
            logger.info(f"Получено {len(spells)} заклинаний")
            return spells
    except Exception as e:
        logger.error(f"Ошибка получения заклинаний: {e}")
        return {"error": f"Не удалось получить заклинания: {str(e)}"}

@tool
async def search_character_by_name(name: str) -> List[Dict[str, Any]]:
    """
    Найти персонажа по имени или части имени.
    
    Args:
        name: Имя персонажа для поиска
    """
    try:
        # Получаем всех персонажей и фильтруем по имени
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HP_API_BASE_URL}/characters")
            response.raise_for_status()
            all_characters = response.json()
            
            # Поиск по имени (нечувствительный к регистру)
            found_characters = [
                char for char in all_characters 
                if name.lower() in char.get('name', '').lower() or
                any(name.lower() in alt_name.lower() for alt_name in char.get('alternate_names', []))
            ]
            
            logger.info(f"Найдено {len(found_characters)} персонажей по запросу '{name}'")
            return found_characters
            
    except Exception as e:
        logger.error(f"Ошибка поиска персонажа {name}: {e}")
        return {"error": f"Не удалось найти персонажа: {str(e)}"}

@tool 
async def search_spells_by_name(name: str) -> List[Dict[str, Any]]:
    """
    Найти заклинание по названию или части названия.
    
    Args:
        name: Название заклинания для поиска
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HP_API_BASE_URL}/spells")
            response.raise_for_status()
            all_spells = response.json()
            
            # Поиск по названию или описанию (нечувствительный к регистру)
            found_spells = [
                spell for spell in all_spells 
                if name.lower() in spell.get('name', '').lower() or
                name.lower() in spell.get('description', '').lower()
            ]
            
            logger.info(f"Найдено {len(found_spells)} заклинаний по запросу '{name}'")
            return found_spells
            
    except Exception as e:
        logger.error(f"Ошибка поиска заклинания {name}: {e}")
        return {"error": f"Не удалось найти заклинание: {str(e)}"}

# Список всех доступных инструментов для экспорта
HP_TOOLS = [
    get_all_characters,
    get_character_by_id,
    get_hogwarts_students,
    get_hogwarts_staff, 
    get_characters_by_house,
    get_all_spells,
    search_character_by_name,
    search_spells_by_name
]