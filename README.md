# LSTM Text Autocomplete

Проект для обучения LSTM модели автодополнения текста на основе твитов с комплексной системой оценки и сравнения с baseline моделями.

## Структура проекта

```
configs/
└── config.yaml             # Конфигурационный файл

data/
├── tweets.txt              # Исходные данные твитов
├── processed_tweets.json   # Обработанные данные
├── train_texts.json        # Обучающие данные
├── val_texts.json          # Валидационные данные
├── test_texts.json         # Тестовые данные
├── model_comparison_results.json    # Результаты сравнения моделей
├── temperature_sensitivity_results.json  # Результаты анализа температуры
└── tuning_results.json     # Результаты подбора параметров

evaluation/
├── __init__.py             # Инициализация модуля оценки
├── baseline_comparison.py  # Сравнение с baseline моделью (DistilGPT2)
├── model_evaluation.py     # Оценка и сравнение моделей
└── parameter_tuning.py     # Подбор оптимальных параметров генерации

models/
├── best_model.pt           # Лучшая модель по ROUGE-L метрике
└── model_epoch_N.pt        # Модели после каждой эпохи

src/
├── __init__.py
├── configs.py              # Конфигурация проекта
├── utils/
│   └── utils.py            # Утилиты обработки данных
├── next_token_dataset.py   # Dataset класс для обучения
├── lstm_model.py           # LSTM модель
├── lstm_train.py           # Функции обучения
└── eval_lstm.py            # Оценка LSTM модели

.gitignore
solution.ipynb              # Основной ноутбук с полным pipeline: обучение, оценка, сравнение
main.py                     # Скрипт обучения без использования ноутбука
inference.py                # Скрипт для инференса
requirements.txt            # Зависимости
```

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Установите PyTorch с поддержкой CUDA:
```bash
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

3. Скачайте данные и создайте необходимые папки:
```bash
# Создайте папку для данных и сохранения весов обученных моделей
mkdir -p data

# Скачайте файл с твитами
curl -o data/tweets.txt https://code.s3.yandex.net/deep-learning/tweets.txt
```

4. Обработанные данные будут созданы автоматически при первом запуске обучения через solution.ipynb или main.py

## Использование

### 1. Обучение с использованием main.py

Запустите обучение с параметрами по умолчанию из `configs/config.yaml`:

```bash
python main.py
```

### 2. Обучение с использованием solution.ipynb

Запустите все ячейки в ноутбуке `solution.ipynb`. Ноутбук содержит полный pipeline:
- Загрузка и предобработка данных
- Обучение LSTM модели
- Сравнение с baseline моделью (DistilGPT2)
- Подбор оптимальных параметров генерации
- Анализ чувствительности к температуре
- Генерация примеров и создание отчетов

### 3. Инференс

Используйте обученную модель для генерации:

```bash
python inference.py
```

Или импортируйте функции в свой код:

```python
from inference import load_model, generate_completion
import torch
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = load_model("models/best_model.pt", tokenizer.vocab_size, device)

# Генерация с параметрами по умолчанию
text = "i am feeling"
completion = generate_completion(text, model, tokenizer, device)
print(f"'{text}' -> '{completion}'")

# Генерация с кастомными параметрами
completion = generate_completion(
    text, model, tokenizer, device,
    max_tokens=15, top_k=10, temperature=0.8
)
```

### 4. Использование модуля evaluation

Модуль evaluation предоставляет инструменты для комплексной оценки моделей:
- Сравнение LSTM с baseline моделью DistilGPT2
- Подбор оптимальных параметров генерации
- Анализ чувствительности к температуре
- Генерация примеров и создание отчетов

**Подробные примеры использования модуля evaluation с результатами и визуализацией представлены в `solution.ipynb`**

## Конфигурация

Все параметры настраиваются в файле `configs/config.yaml`:
Пример:
```yaml
model:
  embed_dim: 256          # Размерность эмбеддингов
  hidden_dim: 256         # Размерность скрытого состояния LSTM
  num_layers: 3           # Количество слоев LSTM

training:
  epochs: 8              # Количество эпох
  lr: 0.0001             # Скорость обучения
  weight_decay: 0.00001  # L2 регуляризация
  batch_size: 32         # Размер батча

data:
  test_size: 0.2         # Доля тестовой выборки
  val_size: 0.5          # Доля валидационной выборки
  random_state: 42       # Seed для воспроизводимости

generation:
  max_new_tokens: 20     # Максимальное количество токенов
  top_k: 20              # Параметр top-k сэмплирования
  temperature: 1.0       # Температура генерации
```

## Особенности

- **Автоматическая обработка данных**: Твиты автоматически обрабатываются и кэшируются
- **Воспроизводимость**: Все случайные процессы используют фиксированный seed
- **Гибкая конфигурация**: Параметры можно изменять через YAML
- **Мониторинг обучения**: Автоматическое сохранение лучшей модели и построение графиков
- **Комплексная оценка**: Сравнение с baseline, подбор параметров, анализ чувствительности
- **ROUGE метрики**: Оценка качества генерации с помощью стандартных метрик

## Результаты

### Модели
Модели сохраняются в папке `models/`:
- `best_model.pt` - лучшая модель по ROUGE-L метрике
- `model_epoch_N.pt` - модели после каждой эпохи

### Данные и результаты
В папке `data/` сохраняются:
- `processed_tweets.json` - предобработанные данные твитов
- `train_texts.json`, `val_texts.json`, `test_texts.json` - разделенные данные
- `model_comparison_results.json` - результаты сравнения LSTM и baseline
- `temperature_sensitivity_results.json` - анализ чувствительности к температуре
- `tuning_results.json` - результаты подбора параметров

### Визуализация
Графики обучения отображаются автоматически и показывают:
- Потери на обучении и валидации
- ROUGE-L метрику на валидации
- Сравнение производительности моделей
- Анализ влияния параметров генерации
