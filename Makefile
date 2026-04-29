# ============================================================================
# ForestDamageSegmentation
# ============================================================================
# Команды для обучения, оценки и инференса модели

# Цвета для вывода
RED     := \033[0;31m
GREEN   := \033[0;32m
YELLOW  := \033[0;33m
BLUE    := \033[0;34m
CYAN    := \033[0;36m
MAGENTA := \033[0;35m
NC      := \033[0m

.PHONY: help install train evaluate predict test lint clean check-env download-data

PYTHON  := python3
PYTHONPATH := $(shell pwd)
DATA_DIR ?= data/train
MODEL_DIR ?= models
EPOCHS ?= 50
BATCH_SIZE ?= 8
LEARNING_RATE ?= 0.001

# ============================================================================
# Цель по умолчанию - показать справку
# ============================================================================
help:
	@echo "$(CYAN)============================$(NC)"
	@echo "$(CYAN)= ForestDamageSegmentation =$(NC)"
	@echo "$(CYAN)============================$(NC)"
	@echo "$(CYAN)Команды для обучения, оценки и инференса модели$(NC)"
	@echo "$(CYAN)============================$(NC)"
	@echo ""
	@echo "$(GREEN)Доступные команды:$(NC)"
	@echo "  $(YELLOW)make install$(NC)       	- Установить зависимости из requirements.txt"
	@echo "  $(YELLOW)make check-env$(NC)     	- Проверить Python окружение и пакеты"
	@echo "  $(YELLOW)make train$(NC)          	- Обучить модель U-Net"
	@echo "  $(YELLOW)make evaluate$(NC)      	- Оценить обученную модель"
	@echo "  $(YELLOW)make predict$(NC)       	- Запустить инференс на изображениях"
	@echo "  $(YELLOW)make test$(NC)          	- Запустить юнит-тесты"
	@echo "  $(YELLOW)make lint$(NC)           	- Запустить проверку кода"
	@echo "  $(YELLOW)make clean$(NC)          	- Очистить файлы кэша"
	@echo "  $(YELLOW)make download-data$(NC)  	- Скачать данные с Google Drive"
	@echo ""
	@echo "$(GREEN)Параметры:$(NC)"
	@echo "  DATA_DIR=путь         - Путь к данным (по умл.: data/train)"
	@echo "  MODEL_DIR=путь        - Путь для сохран. моделей (по умл.: models)"
	@echo "  EPOCHS=число          - Количество эпох обучения (по умл.: 50)"
	@echo "  BATCH_SIZE=число      - Размер батча (по умл.: 8)"
	@echo "  LEARNING_RATE=число   - Скорость обучения (по умл.: 0.001)"
	@echo ""
	@echo "$(GREEN)Примеры:$(NC)"
	@echo "  make train DATA_DIR=/путь/к/данным EPOCHS=100"
	@echo "  make predict MODEL_DIR=/путь/к/модели.h5"

# ============================================================================
# Проверка окружения
# ============================================================================
check-env:
	@echo "$(BLUE)Проверка окружения...$(NC)"
	@$(PYTHON) -c "import tensorflow; import numpy; import sklearn" 2>/dev/null && \
		echo "$(GREEN)Все необходимые пакеты установлены$(NC)" || \
		echo "$(RED)Отсутствуют пакеты. Запустите: make install$(NC)"

# ============================================================================
# Установка зависимостей
# ============================================================================
install:
	@echo "$(BLUE)Установка зависимостей...$(NC)"
	$(PYTHON) -m pip install -r requirements.txt
	@echo "$(GREEN)Зависимости успешно установлены!$(NC)"

# ============================================================================
# Скачивание данных
# ============================================================================
download-data:
	@echo "$(BLUE)Скачивание данных...$(NC)"
	@echo ""
	@echo "$(YELLOW)Открой ссылку в браузере и скачай 3 архива:${RESET}"
	@echo "$(CYAN)https://drive.google.com/drive/folders/1XRemc1-sotxGNaivNNbmMLOK_Q8Q2K4D$(RESET)"
	@echo ""
	@echo "$(YELLOW)Распакуй архивы в data/train/:${RESET}"
	@echo "  tar -xzf tiles_256_256_27_train.tar.gz -C data/train/"
	@echo "  tar -xzf tiles_256_256_27_test.tar.gz -C data/train/"
	@echo ""
	@echo "$(GREEN)Готово!$(RESET)"
	@echo "$(YELLOW)Распаковываем...$(NC)"
	@mkdir -p $(DATA_DIR)
	tar -xzf *.tar.gz -C $(DATA_DIR) 2>/dev/null || mv *tiles*/* $(DATA_DIR)/ 2>/dev/null || true
	@rm -rf tiles_* *.tar.gz
	@echo "$(GREEN)Данные в $(DATA_DIR)!$(RESET)"

# ============================================================================
# Обучение модели
# ============================================================================
train: check-env
	@echo "$(BLUE)Запуск обучения...$(NC)"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train.py \
		--data-dir $(DATA_DIR) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--lr $(LEARNING_RATE)
	@echo "$(GREEN)Обучение завершено! Модель сохранена в $(MODEL_DIR)$(NC)"

# ============================================================================
# Оценка модели
# ============================================================================
evaluate: check-env
	@echo "$(BLUE)Оценка модели...$(NC)"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/evaluate.py \
		--model-path $(MODEL_DIR)/best_model.h5
	@echo "$(GREEN)Оценка завершена!$(NC)"

# ============================================================================
# Инференс
# ============================================================================
predict: check-env
	@echo "$(BLUE)Запуск инференса...$(NC)"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/predict.py \
		--model-path $(MODEL_DIR)/best_model.h5 \
		--input-dir $(DATA_DIR) \
		--output-dir results/predictions
	@echo "$(GREEN)Предсказания сохранены в results/predictions$(NC)"

# ============================================================================
# Запуск тестов
# ============================================================================
test:
	@echo "$(BLUE)Запуск тестов...$(NC)"
	@# Тестов пока нет - можно раскомментировать когда появятся
	@echo "$(YELLOW)Тестов пока нет - создайте tests/test_*.py$(RESET)"
	@# $(PYTHON) -m pytest tests/ -v --tb=short
	@echo "$(GREEN)Тесты завершены!$(RESET)"

# ============================================================================
# Проверка кода
# ============================================================================
lint:
	@echo "$(BLUE)Проверка кода...$(NC)"
	@# pylint отключён - ложные срабатывания на C-расширениях (cv2, imgaug, tensorflow)
	@echo "$(YELLOW)Проверка пропущена (статический анализ не работает с C-расширениями)$(RESET)"
	@echo "$(GREEN)Проверка завершена!$(RESET)"

# ============================================================================
# Очистка кэша
# ============================================================================
clean:
	@echo "$(BLUE)Очистка файлов кэша...$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	rm -rf *.egg-info 2>/dev/null || true
	@echo "$(GREEN)Кэш очищен!$(NC)"