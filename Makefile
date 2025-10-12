.PHONY: demo clean

PORT ?= 8000

## Запуск SPA демо-витрины на локальном сервере
demo:
	@echo "Откройте http://localhost:$(PORT)/frontend_demo/"
	python3 -m http.server $(PORT)

## Очистка временных файлов
clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
