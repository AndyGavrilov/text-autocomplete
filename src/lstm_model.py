"""LSTM Language Model для автодополнения текста.

Этот модуль содержит реализацию LSTM-based языковой модели для генерации
автодополнений текста с поддержкой различных стратегий сэмплирования.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMLanguageModel(nn.Module):
    """LSTM-based языковая модель для автодополнения текста.
    
    Модель использует архитектуру LSTM с embedding слоем для генерации
    продолжений текста на основе входного контекста.
    
    Attributes:
        embedding (nn.Embedding): Слой эмбеддингов для токенов.
        lstm (nn.LSTM): LSTM слой для обработки последовательностей.
        fc (nn.Linear): Полносвязный слой для предсказания следующего токена.
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int = 128, 
        hidden_dim: int = 256, 
        num_layers: int = 2, 
        pad_token_id: int = 0
    ) -> None:
        """Инициализирует LSTM языковую модель.
        
        Args:
            vocab_size: Размер словаря токенизатора.
            embed_dim: Размерность эмбеддингов токенов.
            hidden_dim: Размерность скрытого состояния LSTM.
            num_layers: Количество слоев LSTM.
            pad_token_id: Индекс токена паддинга в словаре.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=0.2, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Прямой проход модели.
        
        Args:
            input_ids: Тензор с индексами токенов формы [batch_size, seq_len].
            hidden: Кортеж (h_0, c_0) скрытых состояний LSTM или None.
            
        Returns:
            Кортеж (logits, hidden), где:
                - logits: Предсказания модели формы [batch_size, seq_len, vocab_size].
                - hidden: Обновленные скрытые состояния LSTM.
        """
        emb = self.embedding(input_ids)               # [batch, seq_len, embed_dim]
        output, hidden = self.lstm(emb, hidden)       # [batch, seq_len, hidden_dim]
        logits = self.fc(output)                      # [batch, seq_len, vocab_size]
        return logits, hidden

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        tokenizer, 
        max_new_tokens: int = 20, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None
    ) -> str:
        """Генерация автодополнения текста.

        Генерирует продолжение входного текста используя различные стратегии
        сэмплирования (top-k, temperature scaling).

        Args:
            input_ids: Входной контекст формы [1, L] с индексами токенов.
            tokenizer: Токенизатор для декодирования токенов в текст.
            max_new_tokens: Максимальное количество генерируемых токенов.
            temperature: Температура для контроля случайности генерации.
                Высокие значения (>1.0) увеличивают разнообразие,
                низкие значения (<1.0) делают генерацию более детерминированной.
            top_k: Количество топ-k токенов для сэмплирования. 
                Если None, используется полное распределение.

        Returns:
            Сгенерированный текст, включающий исходный контекст и дополнение.
            
        Note:
            Модель автоматически переводится в режим eval() для генерации.
        """
        self.eval()
        input_ids = input_ids.to(next(self.parameters()).device)

        # 1) Прогоняем весь контекст для получения корректного hidden состояния
        logits, hidden = self.forward(input_ids)
        last_token = input_ids[:, -1:]
        generated = input_ids.clone()

        # 2) Автодополнение токен за токеном
        for _ in range(max_new_tokens):
            logits, hidden = self.forward(last_token, hidden)   # [1,1,V]
            logits = logits[:, -1, :] / max(1e-8, temperature)

            if top_k is not None:
                vals, idxs = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(vals, dim=-1)
                next_token = idxs.gather(-1, torch.multinomial(probs, 1))
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_token], dim=1)
            last_token = next_token

        return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


    @torch.no_grad()
    def generate_tokens(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 20, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """Генерация токенов без декодирования в текст.
        
        Аналогично методу generate(), но возвращает индексы токенов вместо
        декодированного текста. Полезно для дальнейшей обработки или анализа.

        Args:
            input_ids: Входной контекст формы [1, L] с индексами токенов.
            max_new_tokens: Максимальное количество генерируемых токенов.
            temperature: Температура для контроля случайности генерации.
            top_k: Количество топ-k токенов для сэмплирования.

        Returns:
            Тензор с полной последовательностью [1, L + max_new_tokens],
            включающей исходный контекст и сгенерированные токены.
            
        Note:
            Модель автоматически переводится в режим eval() для генерации.
        """
        self.eval()
        input_ids = input_ids.to(next(self.parameters()).device)
        
        # Прогоняем контекст для получения hidden состояния
        logits, hidden = self.forward(input_ids)
        last_token = input_ids[:, -1:]
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            logits, hidden = self.forward(last_token, hidden)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            
            if top_k is not None:
                vals, idxs = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(vals, dim=-1)
                next_token = idxs.gather(-1, torch.multinomial(probs, 1))
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            last_token = next_token
        
        return generated