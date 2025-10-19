.PHONY: ui api dev clean

API_HOST ?= 127.0.0.1
API_PORT ?= 8000

## Запуск UI (SPA) вместе с API
ui:
	@echo "UI и API доступны на http://$(API_HOST):$(API_PORT)"
	python3 -m server --host $(API_HOST) --port $(API_PORT)

## Запуск API сервера (Flask)
api:
	@echo "API доступно на http://$(API_HOST):$(API_PORT)/api"
	python3 -m server --host $(API_HOST) --port $(API_PORT)

## Одновременный запуск фронта и API (для локальной разработки)
dev:
	@echo "Запускаем Flask-сервер с UI и API..."
	python3 -m server --host $(API_HOST) --port $(API_PORT) --debug

## Очистка временных файлов
clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
