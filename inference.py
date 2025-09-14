import torch
from transformers import AutoTokenizer
from src.lstm_model import LSTMLanguageModel
from src.configs import CONFIG, TOKENIZER

def load_model(model_path, vocab_size, device):
    """Загружает обученную модель из конфигурации."""
    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embed_dim=CONFIG['model']['embed_dim'],
        hidden_dim=CONFIG['model']['hidden_dim'],
        num_layers=CONFIG['model']['num_layers'],
        pad_token_id=0
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def generate_completion(text, model, tokenizer, device, max_tokens=None, top_k=None, temperature=None):
    """Генерирует автодополнение для текста.
    
    Args:
        text: Входной текст для автодополнения
        model: Обученная LSTM модель
        tokenizer: Токенизатор
        device: Устройство для вычислений
        max_tokens: Максимальное количество токенов (по умолчанию из CONFIG)
        top_k: Параметр top-k сэмплирования (по умолчанию из CONFIG)
        temperature: Температура генерации (по умолчанию из CONFIG)
    
    Returns:
        Сгенерированный текст
    """
    if max_tokens is None:
        max_tokens = CONFIG['generation']['max_new_tokens']
    if top_k is None:
        top_k = CONFIG['generation']['top_k']
    if temperature is None:
        temperature = CONFIG['generation']['temperature']
        
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    generated = model.generate(
        input_ids, tokenizer, 
        max_new_tokens=max_tokens, 
        top_k=top_k,
        temperature=temperature
    )
    return generated

if __name__ == "__main__":
    # Пример использования
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("models/best_model.pt", TOKENIZER.vocab_size, device)
    
    # Примеры генерации с разными параметрами
    text = "i am feeling"
    
    # Используем параметры по умолчанию из CONFIG
    completion_default = generate_completion(text, model, TOKENIZER, device)
    print(f"По умолчанию: '{text}' -> '{completion_default}'")
    
    # Переопределяем параметры
    completion_custom = generate_completion(
        text, model, TOKENIZER, device, 
        max_tokens=15, top_k=10, temperature=0.8
    )
    print(f"Кастомные параметры: '{text}' -> '{completion_custom}'")