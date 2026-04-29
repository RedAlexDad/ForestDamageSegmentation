# ForestDamageSegmentation

Neural network for satellite image segmentation to detect forest damage.

## MLflow Integration

Проект использует **MLflow** для:
- Логирование экспериментов
- Визуализация метрик (графики loss, dice, IoU)
- Сохранение артефактов (модели, изображения, конфиги)
- Сравнение запусков
- Регистрация моделей (Model Registry)

### Запуск MLflow

```bash
# Локальный сервер
mlflow server --backend-store-uri ./mlruns --default-artifact-root ./mlruns/artifacts

# UI доступен на http://localhost:5000
```

### Конфигурация MLflow

```python
import mlflow
from mlflow.tracking import MlflowClient

# Настройка
mlflow.set_experiment("forest-damage-segmentation")
mlflow.tensorflow.autolog()  # Автоматическое логирование TF

# Ручное логирование
mlflow.log_params({"learning_rate": 0.001, "batch_size": 8})
mlflow.log_metrics({"dice_coefficient": 0.85, "val_loss": 0.15})
mlflow.log_artifact("model.h5")
```

---

## Декомпозиционный план

### Фаза 1: Подготовка данных (3 дня)

- [ ] 1.1 Загрузка данных с Google Drive
- [ ] 1.2 Распаковка архивов
- [ ] 1.3 Проверка структуры данных
- [ ] 1.4 Разделение на train/valid/test
- [ ] 1.5 Аугментация данных
- [ ] 1.6 Создание генераторов данных
- [ ] 1.7 Визуализация семплов (до/после аугментации)
- [ ] 1.8 Статистика датасета (гистограммы, распределения)
- [ ] 1.9 Логирование в MLflow

### Фаза 2: Модель U-Net (5 дней)

- [ ] 2.1 Архитектура U-Net
- [ ] 2.2 Encoder (CNN)
- [ ] 2.3 Decoder (Upsampling)
- [ ] 2.4 Skip connections
- [ ] 2.5 Компиляция модели
- [ ] 2.6 Обучение модели
- [ ] 2.7 Валидация
- [ ] 2.8 Сохранение лучшей модели
- [ ] 2.9 Логирование архитектуры в MLflow

### Фаза 3: Метрики и функции потерь (2 дня)

- [ ] 3.1 Dice Coefficient
- [ ] 3.2 IoU (Intersection over Union)
- [ ] 3.3 Binary Cross-Entropy
- [ ] 3.4 Tversky Loss
- [ ] 3.5 Комбинированные функции потерь
- [ ] 3.6 Логирование метрик в MLflow (графики)

### Фаза 4: Ансамбли моделей (3 дня)

- [ ] 4.1 Average Ensemble
- [ ] 4.2 Weighted Average Ensemble
- [ ] 4.3 Stacking Ensemble
- [ ] 4.4 Сравнение ансамблей
- [ ] 4.5 Выбор лучшего ансамбля
- [ ] 4.6 Логирование экспериментов в MLflow

### Фаза 5: Инференс и предикты (2 дня)

- [ ] 5.1 Загрузка обученных моделей
- [ ] 5.2 Предикт на тестовых данных
- [ ] 5.3 Постобработка (фильтрация, морфология)
- [ ] 5.4 Сохранение масок
- [ ] 5.5 Визуализация результатов
- [ ] 5.6 Логирование предиктов как артефакты

### Фаза 6: Тестирование и оптимизация (2 дня)

- [ ] 6.1 Тюнинг гиперпараметров
- [ ] 6.2 Тестирование на разных данных
- [ ] 6.3 Анализ ошибок
- [ ] 6.4 Оптимизация скорости
- [ ] 6.5 Экспорт модели
- [ ] 6.6 Сравнение экспериментов в MLflow

### Фаза 7: Документация и отчёт (1 день)

- [ ] 7.1 README
- [ ] 7.2 Описание архитектуры
- [ ] 7.3 Инструкция по запуску
- [ ] 7.4 Примеры использования
- [ ] 7.5 Экспорт отчёта из MLflow

---

## Что логировать в MLflow

### Параметры (Parameters)
```python
mlflow.log_params({
    "learning_rate": 0.001,
    "batch_size": 8,
    "epochs": 20,
    "image_size": 256,
    "n_filters": 64,
    "loss_function": "dice_loss",
    "optimizer": "adam",
    "augmentation": True,
    "model_name": "UNet_v1"
})
```

### Метрики (Metrics) - каждую эпоху
```python
mlflow.log_metrics({
    "train_loss": 0.15,
    "train_dice": 0.85,
    "train_iou": 0.72,
    "val_loss": 0.18,
    "val_dice": 0.82,
    "val_iou": 0.70,
    "epoch": 10,
    "learning_rate": 0.001,
    "batch_time": 0.35,
    "gpu_memory_mb": 2048
})
```

### Артефакты (Artifacts)
```python
# Модель
mlflow.keras.log_model(model, "model")

# Графики
mlflow.log_figure(fig, "training_curves.png")

# Изображения
mlflow.log_image(image, "sample_prediction.png")

# Конфиги
mlflow.log_artifact("config.yaml")

# Чекпоинты
mlflow.log_model(checkpoint, "checkpoint")
```

### Графики для визуализации

1. **Training Curves**
   - Loss (train/valid) по эпохам
   - Dice coefficient по эпохам
   - IoU по эпохам

2. **Data Visualization**
   - Семплы изображений с масками
   - Распределение классов
   - Аугментации (до/после)

3. **Predictions**
   - Входное изображение
   - Ground truth маска
   - Предсказанная маска
   - Overlay (наложение)

4. **Confusion Matrix**
   - TP, TN, FP, FN

5. **PR Curve**
   - Precision-Recall curve

6. **Hyperparameter Importance**
   - Какие гиперпараметры важнее

---

## Структура проекта

```
ForestDamageSegmentation/
├── data/                     # Данные
│   ├── raw/                 # Сырые данные
│   ├── processed/           # Обработанные данные
│   ├── train/              # Обучающая выборка
│   ├── valid/              # Валидационная выборка
│   └── test/               # Тестовая выборка
│
├── src/                     # Исходный код
│   ├── data/                # Загрузка и preprocessing данных
│   │   ├── __init__.py
│   │   ├── dataset.py       # Датасеты и генераторы
│   │   ├── augmentation.py # Аугментация
│   │   └── loader.py        # Загрузчик данных
│   │
│   ├── models/             # Модели
│   │   ├── __init__.py
│   │   ├── unet.py         # U-Net архитектура
│   │   └── layers.py      # Кастомные слои
│   │
│   ├── metrics/            # Метрики
│   │   ├── __init__.py
│   │   ├── dice.py        # Dice coefficient
│   │   └── iou.py         # IoU
│   │
│   ├── losses/            # Функции потерь
│   │   ├── __init__.py
│   │   ├── dice_loss.py
│   │   ├── tversky_loss.py
│   │   └── combined_loss.py
│   │
│   ├── training/           # Обучение
│   │   ├── __init__.py
│   │   ├── trainer.py      # Класс для обучения
│   │   └── callbacks.py    # Callback функции
│   │
│   ├── ensembles/          # Ансамбли
│   │   ├── __init__.py
│   │   ├── average.py      # Average ensemble
│   │   ├── weighted.py     # Weighted ensemble
│   │   └── stacking.py     # Stacking ensemble
│   │
│   ├── inference/          # Инференс
│   │   ├── __init__.py
│   │   ├── predictor.py    # Класс для предикта
│   │   └── postprocess.py   # Постобработка
│   │
│   ├── utils/             # Утилиты
│   │   ├── __init__.py
│   │   ├── helpers.py
│   │   ├── config.py
│   │   └── visualizations.py  # Визуализации
│   │
│   └── mlflow_utils/       # MLflow утилиты
│       ├── __init__.py
│       ├── logging.py      # Логирование
│       └── tracking.py    # Отслеживание
│
├── notebooks/              # Jupyter ноутбуки (для исследования)
│   ├── eda.ipynb          # Exploratory Data Analysis
│   ├── experiments.ipynb   # Эксперименты
│   └── model_analysis.ipynb  # Анализ моделей
│
├── tests/                 # Тесты
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_losses.py
│
├── configs/               # Конфигурации
│   ├── config.yaml
│   └── model_config.yaml
│
├── logs/                   # Логи и чекпоинты
│   ├── experiments/
│   └── checkpoints/
│
├── mlruns/                 # MLflow логи (создаётся автоматически)
│
├── results/               # Результаты
│   ├── predictions/       # Предикты
│   └── metrics/          # Метрики
│
├── scripts/              # Скрипты запуска
│   ├── train.py          # Обучение модели
│   ├── evaluate.py       # Оценка модели
│   ├── predict.py        # Предикт
│   ├── download_data.py  # Скачивание данных
│   └── run_experiment.py  # Запуск эксперимента
│
├── requirements.txt      # Зависимости
├── setup.py              # Установка пакета
├── pyproject.toml        # Конфиг проекта
├── .gitignore
├── LICENSE
└── README.md
```

---

## Основные классы и функции

### `src/mlflow_utils/logging.py`

```python
import mlflow
from mlflow.tracking import MlflowClient

class MLflowLogger:
    """Класс для логирования в MLflow"""
    
    def __init__(self, experiment_name, tracking_uri=None):
        self.experiment_name = experiment_name
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def log_params(self, params):
        mlflow.log_params(params)
    
    def log_metrics(self, metrics, step=None):
        mlflow.log_metrics(metrics, step=step)
    
    def log_figure(self, fig, name):
        mlflow.log_figure(fig, name)
    
    def log_image(self, image, name):
        mlflow.log_image(image, name)
    
    def log_model(self, model, name):
        mlflow.keras.log_model(model, name)
    
    def log_artifact(self, path):
        mlflow.log_artifact(path)
    
    def start_run(self, run_name=None, tags=None):
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def end_run(self, status=None):
        mlflow.end_run(status=status)
```

### `src/training/trainer.py`

```python
class ModelTrainer:
    """Класс для обучения модели с MLflow логированием"""
    
    def __init__(self, model, train_dataset, valid_dataset, config, logger=None):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.config = config
        self.logger = logger  # MLflowLogger
    
    def train(self, epochs, callbacks):
        for epoch in range(epochs):
            # Обучение
            train_metrics = self.train_epoch()
            
            # Валидация
            val_metrics = self.validate()
            
            # Логирование
            if self.logger:
                self.logger.log_metrics({
                    **train_metrics,
                    **val_metrics
                }, step=epoch)
```

---

## Workflow с MLflow

### Запуск эксперимента

```python
from src.mlflow_utils import MLflowLogger
from src.data import ForestDataset
from src.models import build_unet
from src.training import ModelTrainer
from src.utils import plot_training_curves

# Конфиг
config = {
    "learning_rate": 0.001,
    "batch_size": 8,
    "epochs": 20,
    "image_size": 256
}

# MLflow
logger = MLflowLogger("forest-damage-segmentation")

# Логирование параметров
logger.log_params(config)

# Запуск эксперимента
with logger.start_run("unet-v1-run-001"):
    # Данные
    train_ds = ForestDataset(...)
    valid_ds = ForestDataset(...)
    
    # Модель
    model = build_unet(...)
    
    # Обучение
    trainer = ModelTrainer(model, train_ds, valid_ds, config, logger)
    trainer.train(epochs=20)
    
    # Логирование финальных метрик
    logger.log_metrics({
        "best_dice": 0.85,
        "best_iou": 0.72
    })
    
    # Логирование модели
    logger.log_model(model, "final_model")
    
    # Логирование графиков
    fig = plot_training_curves(history)
    logger.log_figure(fig, "training_curves.png")
```

### Визуализация в MLflow UI

1. **Metrics** - графики loss, dice, IoU по эпохам
2. **Artifacts** - модели, изображения, конфиги
3. **Comparisons** - сравнение разных запусков
4. **Parameters** - гиперпараметры

---

## Checklist прогресса с MLflow

| Фаза | Задача | Статус | MLflow логи | Дата завершения |
|------|--------|--------|------------|-------------|
| 1 | Подготовка данных | [ ] | params, artifacts | |
| 2 | U-Net модель | [ ] | model, params | |
| 3 | Метрики | [ ] | metrics | |
| 4 | Ансамбли | [ ] | metrics, comparison | |
| 5 | Инференс | [ ] | artifacts | |
| 6 | Оптимизация | [ ] | comparison | |
| 7 | Документация | [ ] | - | |

---

## Зависимости

```
tensorflow>=2.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-image>=0.21.0
tifffile>=2023.7.10
matplotlib>=3.7.0
albumentations>=1.3.0
pyyaml>=6.0
tqdm>=4.65.0
pillow>=10.0.0
opencv-python>=4.8.0
mlflow>=2.10.0
```

---

## TODO

- [ ] Настроить MLflow
- [ ] Создать MLflowLogger класс
- [ ] Интегрировать логирование в training
- [ ] Логировать графики
- [ ] Логировать артефакты
- [ ] Настроить Model Registry
- [ ] Добавить визуализации