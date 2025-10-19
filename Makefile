.PHONY: ui api dev clean

UI_PORT ?= 5173
API_HOST ?= 127.0.0.1
API_PORT ?= 8000

## Запуск SPA демо-витрины (только фронтенд)
ui:
	@echo "Откройте http://localhost:$(UI_PORT)/ для доступа к фронтенду"
	python3 -m http.server $(UI_PORT) --directory frontend_demo

## Запуск API сервера (Flask)
api:
	@echo "API доступно на http://$(API_HOST):$(API_PORT)/api"
	python3 -m server --host $(API_HOST) --port $(API_PORT)

## Одновременный запуск фронта и API (для локальной разработки)
dev:
	@echo "Запускаем API и статику..."
	@bash -c 'python3 -m server --host $(API_HOST) --port $(API_PORT) --debug & \
	API_PID=$$!; \
	trap "kill $$API_PID" EXIT; \
	echo "Фронт доступен на http://localhost:$(UI_PORT)/"; \
	python3 -m http.server $(UI_PORT) --directory frontend_demo'

## Очистка временных файлов
clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
