import torch
from typing import Dict, Any, List, Union
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.configs import ROUGE


def evaluate_rouge(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    device: torch.device,
    pad_token_id: int = 0,
    max_samples: int = 2000
) -> Dict[str, float]:
    """Вычисляет метрики ROUGE для модели автодополнения текста.
    
    Функция оценивает качество генерации текста модели, сравнивая сгенерированные
    продолжения с эталонными текстами с помощью метрик ROUGE (Recall-Oriented
    Understudy for Gisting Evaluation).
    
    Args:
        model: Обученная модель для генерации текста.
        dataloader: DataLoader с данными для оценки.
        tokenizer: Токенизатор.
        device: Устройство для вычислений (CPU или GPU).
        pad_token_id: ID токена паддинга. По умолчанию 0.
        max_samples: Максимальное количество образцов для оценки. По умолчанию 2000.
        
    Returns:
        Словарь с метриками ROUGE, содержащий значения для ROUGE-1, ROUGE-2, ROUGE-L
        и их вариаций (precision, recall, fmeasure).
        
    Note:
        Функция использует стратегию генерации, где:
        - Берется 75% от исходной последовательности как контекст
        - Генерируется продолжение для оставшихся 25%
        - Сравнивается сгенерированный текст с эталонным продолжением
    """
    model.eval()
    preds, refs = [], []
    processed = 0

    with torch.no_grad():
        for batch in dataloader:
            if processed >= max_samples:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            lengths = (input_ids != pad_token_id).sum(dim=1).tolist()
            for i, L in enumerate(lengths):
                if processed >= max_samples:
                    break
                if L < 4:
                    continue
                cutoff = max(1, int(L * 0.75))
                context = input_ids[i:i+1, :cutoff]
                target_len = L - cutoff
                if target_len <= 0:
                    continue

                # Генерируем токены
                generated_ids = model.generate_tokens(context, max_new_tokens=target_len, temperature=1.0, top_k=20)
                
                # Берём только новые токены (хвост)
                new_tokens = generated_ids[0, cutoff:].tolist()
                ref_tokens = labels[i, cutoff:L].tolist()
                
                # Убираем паддинги
                new_tokens = [t for t in new_tokens if t != pad_token_id]
                ref_tokens = [t for t in ref_tokens if t != pad_token_id]
                
                # Декодируем для ROUGE
                ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
                pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                preds.append(pred_text)
                refs.append(ref_text)
                processed += 1

    return ROUGE.compute(predictions=preds, references=refs)