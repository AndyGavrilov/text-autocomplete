import logging
import random
import torch
from transformers import AutoTokenizer

from src.configs import DATA_PATH, PROJECT_PATH, CONFIG, TOKENIZER
from src.utils.utils import process_tweets_dataset, split_data
from src.next_token_dataset import LMTextDataset, collate_fn
from src.lstm_model import LSTMLanguageModel
from src.lstm_train import train_model
from src.eval_lstm import evaluate_rouge
from torch.utils.data import DataLoader

def main():
    """Основная функция для обучения LSTM модели автодополнения текста."""
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Настройка устройства и воспроизводимости
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(CONFIG['data']['random_state'])
    torch.manual_seed(CONFIG['data']['random_state'])
    
    logger.info(f"Используется устройство: {device}")
    logger.info(f"Конфигурация: {CONFIG}")
    
    # Загрузка токенизатора
    logger.info("Загрузка токенизатора...")
    
    # Обработка данных
    logger.info("Обработка данных...")
    processed_texts = process_tweets_dataset(tokenizer=TOKENIZER)
    
    # Разделение на train/val/test
    logger.info("Разделение данных...")
    train_texts, val_texts, test_texts = split_data(
        processed_texts,
        test_size=CONFIG['data']['test_size'],
        val_size=CONFIG['data']['val_size'],
        random_state=CONFIG['data']['random_state']
    )
    
    # Создание датасетов и даталоадеров
    logger.info("Создание датасетов...")
    train_dataset = LMTextDataset(train_texts, TOKENIZER)
    val_dataset = LMTextDataset(val_texts, TOKENIZER)
    test_dataset = LMTextDataset(test_texts, TOKENIZER)
    
    pad_token_id = TOKENIZER.pad_token_id or 0
    batch_size = CONFIG['training']['batch_size']
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        collate_fn=lambda b: collate_fn(b, pad_token_id)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        collate_fn=lambda b: collate_fn(b, pad_token_id)
    )
    
    # Создание модели
    logger.info("Создание модели...")
    vocab_size = TOKENIZER.vocab_size
    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embed_dim=CONFIG['model']['embed_dim'],
        hidden_dim=CONFIG['model']['hidden_dim'],
        num_layers=CONFIG['model']['num_layers'],
        pad_token_id=pad_token_id
    )
    
    # Обучение модели
    logger.info("Начало обучения...")
    trained_model, best_path = train_model(
        model, train_loader, val_loader, TOKENIZER, device,
        epochs=CONFIG['training']['epochs'], 
        lr=CONFIG['training']['lr'], 
        weight_decay=CONFIG['training']['weight_decay'], 
        pad_token_id=pad_token_id
    )
    
    # Загрузка лучшей модели для тестирования
    logger.info("Загрузка лучшей модели...")
    best_model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embed_dim=CONFIG['model']['embed_dim'],
        hidden_dim=CONFIG['model']['hidden_dim'],
        num_layers=CONFIG['model']['num_layers'],
        pad_token_id=pad_token_id
    ).to(device)
    best_model.load_state_dict(torch.load(best_path))
    
    # Тестирование модели
    logger.info("Тестирование модели...")
    sample_text = "i am feeling"
    input_ids = TOKENIZER.encode(sample_text, return_tensors="pt").to(device)
    generated = best_model.generate(
        input_ids, TOKENIZER, 
        max_new_tokens=CONFIG['generation']['max_new_tokens'], 
        top_k=CONFIG['generation']['top_k']
    )
    logger.info(f"Пример генерации: '{sample_text}' -> '{generated}'")
    
    # Оценка на тестовой выборке
    logger.info("Оценка на тестовой выборке...")
    test_rouge = evaluate_rouge(best_model, test_loader, TOKENIZER, device)
    logger.info(f"Тестовые метрики ROUGE: {test_rouge}")
    
    logger.info("Обучение завершено!")

if __name__ == "__main__":
    main()