import json
import logging
import os
import re
from typing import List, Optional, Tuple, Union

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm

from src.configs import DATA_PATH, PROJECT_PATH

# Настройка логгера
logger = logging.getLogger(__name__)

def preprocess_text(text: str, tokenizer: AutoTokenizer=None) -> Union[str, List[str]]:
    """
    Предобработка текста для модели автодополнения.
    
    Args:
        text: Исходный текст для обработки
        tokenizer: Токенизатор для разбиения текста на токены
        
    Returns:
        Обработанный текст или список токенов, если передан токенизатор
    """
    # 1. lower-case
    text = text.lower()
    
    # 2. замена url и mentions
    text = re.sub(r'http\S+|www\.\S+', '<url>', text)
    text = re.sub(r'@\w+', '<user>', text)
    
    # 3. хештеги -> слово без решётки
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # 4. удаление эмоджи (замена на токен)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('<emoji>', text)
    
    # 5. нормализация whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. оставить только ascii + базовая пунктуация
    text = re.sub(r'[^a-z0-9\s\.\,\!\?\;\:\'\"\<\>\_]', '', text)
    
    # 7. токенизация
    if tokenizer:
        return tokenizer.tokenize(text)
    return text


def load_tweets(tweets_path: str = DATA_PATH / 'tweets.txt') -> List[str]:
    """Загружает сырые твиты из файла.
    
    Args:
        tweets_path: Путь к файлу с твитами
        
    Returns:
        Список сырых твитов
        
    Raises:
        FileNotFoundError: Если файл с твитами не найден
    """
    if not os.path.exists(tweets_path):
        raise FileNotFoundError(f"Файл с твитами не найден: {tweets_path}")
    
    logger.info(f"Загружаем твиты из {tweets_path}")
    with open(tweets_path, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    
    logger.info(f"Загружено {len(tweets)} твитов")
    return tweets


def process_tweets(tweets: List[str], tokenizer=None) -> List[Union[str, List[str]]]:
    """Обрабатывает список твитов.
    
    Args:
        tweets: Список сырых твитов
        tokenizer: Токенизатор для обработки текстов
        
    Returns:
        Список обработанных текстов/токенов
    """
    logger.info("Обрабатываем тексты...")
    processed_texts = []
    for tweet in tqdm(tweets, desc="Обработка твитов"):
        processed_text = preprocess_text(tweet, tokenizer=tokenizer)
        processed_texts.append(processed_text)
    
    logger.info(f"Обработка завершена. Получено {len(processed_texts)} обработанных текстов")
    return processed_texts


def save_processed_data(data: List, path: str) -> None:
    """Сохраняет обработанные данные в файл.
    
    Args:
        data: Список обработанных текстов/токенов
        path: Путь для сохранения данных
    """
    logger.info(f"Сохраняем данные в {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    logger.info("Данные успешно сохранены")


def process_tweets_dataset(
    tweets_path: str = DATA_PATH / 'tweets.txt',
    processed_path: str = DATA_PATH / 'processed_tweets.json',
    tokenizer=None,
    force_reprocess: bool = False,
    save_processed: bool = True
) -> List[Union[str, List[str]]]:
    """
    Универсальная функция-обертка для полного цикла обработки твитов.
    
    Загружает исходные твиты, обрабатывает их и сохраняет результат.
    Если обработанные данные уже существуют и force_reprocess=False,
    загружает их из файла.
    
    Args:
        tweets_path: Путь к файлу с исходными твитами
        processed_path: Путь для сохранения обработанных данных
        tokenizer: Токенизатор для обработки текстов
        force_reprocess: Принудительная переобработка, даже если файл существует
        save_processed: Сохранять ли обработанные данные в файл
        
    Returns:
        Список обработанных текстов/токенов
        
    Raises:
        FileNotFoundError: Если файл с твитами не найден
        ValueError: Если не удалось загрузить или сохранить данные
    """
    # Проверяем кэш
    if os.path.exists(processed_path) and not force_reprocess:
        logger.info(f"Загружаем обработанные данные из {processed_path}")
        try:
            with open(processed_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Ошибка при загрузке: {e}")
            logger.info("Переходим к переобработке...")
    
    # Загружаем и обрабатываем
    tweets = load_tweets(tweets_path)
    processed_texts = process_tweets(tweets, tokenizer)
    
    if save_processed:
        save_processed_data(processed_texts, processed_path)
    
    return processed_texts


def split_data(
    processed_texts: List[Union[str, List[str]]],
    force_reprocess: bool = False,
    save_data: bool = True,
    test_size: float = 0.2,
    val_size: float = 0.5,
    random_state: int = 42
) -> Tuple[List[Union[str, List[str]]], List[Union[str, List[str]]], List[Union[str, List[str]]]]:
    """
    Разделяет обработанные тексты на train/val/test выборки.
    
    Args:
        processed_texts: Список обработанных текстов/токенов
        force_reprocess: Принудительная переобработка, даже если файлы существуют
        save_data: Сохранять ли разделенные данные в файлы
        test_size: Доля test выборки от общего объема данных
        val_size: Доля val выборки от оставшихся после выделения test данных
        random_state: Seed для воспроизводимости разделения
        
    Returns:
        Кортеж из трех списков: (train_texts, val_texts, test_texts)
    """
    # Пути для сохранения
    train_path = DATA_PATH / 'train_texts.json'
    val_path = DATA_PATH / 'val_texts.json'
    test_path = DATA_PATH / 'test_texts.json'
    
    # Проверяем кэш
    if (os.path.exists(train_path) and os.path.exists(val_path) and 
        os.path.exists(test_path) and not force_reprocess):
        logger.info("Загружаем разделенные данные из файлов")
        try:
            with open(train_path, 'r', encoding='utf-8') as file:
                train_texts = json.load(file)
            with open(val_path, 'r', encoding='utf-8') as file:
                val_texts = json.load(file)
            with open(test_path, 'r', encoding='utf-8') as file:
                test_texts = json.load(file)
            
            logger.info(f"Загружено: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")
            return train_texts, val_texts, test_texts
            
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Ошибка при загрузке разделенных данных: {e}")
            logger.info("Переходим к переразделению...")
    
    # Выполняем разделение
    logger.info("Разделяем данные на train/val/test выборки")
    train_texts, temp_texts = train_test_split(
        processed_texts, 
        test_size=test_size, 
        random_state=random_state
    )
    val_texts, test_texts = train_test_split(
        temp_texts, 
        test_size=val_size, 
        random_state=random_state
    )
    
    logger.info(f"Разделение завершено: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")
    
    # Сохраняем данные если требуется
    if save_data:
        logger.info("Сохраняем разделенные данные")
        save_processed_data(train_texts, train_path)
        save_processed_data(val_texts, val_path)
        save_processed_data(test_texts, test_path)
        logger.info("Разделенные данные успешно сохранены")
    
    return train_texts, val_texts, test_texts