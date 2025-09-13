import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Optional


class LMTextDataset(Dataset):
    """
    Dataset для обучения языковой модели на задаче предсказания следующего токена.
    
    Для последовательности токенов [A, B, C, D] создает:
    - input_ids: [A, B, C] (все кроме последнего)
    - labels: [B, C, D] (все кроме первого, сдвинутые на 1)
    """

    def __init__(self, texts: List[List[str]], tokenizer):
        """
        Args:
            texts: Список текстов, где каждый текст - список токенов
            tokenizer: Токенизатор для преобразования токенов в ID
        """
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Возвращает количество примеров в датасете.
        
        Returns:
            Количество примеров
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Возвращает один пример для обучения.
        
        Args:
            idx: Индекс примера
            
        Returns:
            Словарь с ключами 'input_ids' и 'labels'
        """
        tokens = self.texts[idx]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # X (все кроме последнего), Y (все кроме первого)
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Функция для группировки примеров в батч с динамическим паддингом.
    
    Группирует примеры по диапазонам длины для эффективного паддинга.
    
    Args:
        batch: Список примеров из датасета
        pad_token_id: ID токена для паддинга
        
    Returns:
        Словарь с паддинговыми батчами 'input_ids' и 'labels'
    """
    def get_length_group(item: Dict[str, torch.Tensor]) -> int:
        """
        Определяет группу длины для элемента.
        
        Args:
            item: Пример из датасета
            
        Returns:
            Группа длины
        """
        length = len(item["input_ids"])
        if length <= 10:
            return 0      # Короткие последовательности
        elif length <= 20:
            return 1      # Средние последовательности  
        elif length <= 40:
            return 2      # Длинные последовательности
        else:
            return 3      # Очень длинные последовательности
    
    # Сортируем: сначала по группе, потом по длине внутри группы
    batch.sort(key=lambda x: (get_length_group(x), len(x["input_ids"])), reverse=True)
    
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=pad_token_id)

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded
    }