import logging
import os
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tqdm import tqdm

from src.configs import PROJECT_PATH
from src.eval_lstm import evaluate_rouge

# Настройка логгера
logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: AutoTokenizer,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    pad_token_id: int = 0,
    save_dir: Union[str, Path] = PROJECT_PATH / "models"
) -> Tuple[nn.Module, str]:
    """Обучает LSTM модель для автодополнения текста.
    
    Функция выполняет полный цикл обучения модели, включая:
    - Обучение на тренировочных данных
    - Валидацию на каждом эпохе с вычислением loss и ROUGE метрик
    - Сохранение лучшей модели по метрике ROUGE-L
    - Визуализацию процесса обучения через графики
    
    Args:
        model: LSTM модель для обучения
        train_loader: DataLoader с тренировочными данными
        val_loader: DataLoader с валидационными данными
        tokenizer: Токенизатор
        device: Устройство для вычислений (CPU/GPU)
        epochs: Количество эпох обучения (по умолчанию 3)
        lr: Скорость обучения (по умолчанию 1e-3)
        weight_decay: Коэффициент регуляризации L2 (по умолчанию 0.0)
        pad_token_id: ID токена паддинга (по умолчанию 0)
        save_dir: Путь к директории для сохранения моделей (str или Path, по умолчанию PROJECT_PATH / "models")
        
    Returns:
        Tuple[nn.Module, str]: Кортеж из обученной модели и пути к лучшей модели
        
    Note:
        - Использует градиентное обрезание для стабильности обучения
        - Сохраняет модель после каждой эпохи
        - Автоматически очищает GPU память для предотвращения переполнения
        - Строит графики потерь и ROUGE метрик
    """
                
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    model = model.to(device)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_rouge = 0.0
    best_model_path = save_dir / "best_model.pt"
    best_model_epoch = None

    # Добавляем списки для графиков
    train_losses = []
    val_losses = []
    val_rouges = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids)

            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                logits, _ = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Очистка перед ROUGE
        torch.cuda.empty_cache()

        # ROUGE на валидации
        val_rouge_score = evaluate_rouge(
            model=model,
            dataloader=val_loader,
            tokenizer=tokenizer,
            device=device
        )
        # Универсальное извлечение ROUGE-L
        v = val_rouge_score["rougeL"]
        val_rouge_l = v.mid.fmeasure if hasattr(v, "mid") else float(v)
        val_rouges.append(val_rouge_l)

        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val ROUGE-L={val_rouge_l:.4f}")

        # Очистка памяти после эпохи
        torch.cuda.empty_cache()

        # Сохранение лучшей модели
        if val_rouge_l > best_rouge:
            best_rouge = val_rouge_l
            torch.save(model.state_dict(), best_model_path)
            best_model_epoch = epoch+1
            logger.info(f"Сохранена новая лучшая модель (ROUGE-L={best_rouge:.4f}), эпоха {best_model_epoch}")

        # Сохранение модели каждой эпохи
        epoch_path = save_dir / f"model_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), epoch_path)
        logger.info(f"Сохранена модель эпохи {epoch+1}")

    logger.info(f"Обучение завершено. Лучший ROUGE-L={best_rouge:.4f}")
    logger.info(f"Лучшая модель сохранена в {best_model_path}, эпоха {best_model_epoch}")


    # Графики
    plt.figure(figsize=(15, 5))
    
    # Создаем массив номеров эпох (начинаем с 1)
    epochs_range = range(1, epochs + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_rouges, 'r-', label='Val ROUGE-L')
    plt.xlabel('Epoch')
    plt.ylabel('ROUGE-L')
    plt.title('Validation ROUGE-L')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    return model, best_model_path