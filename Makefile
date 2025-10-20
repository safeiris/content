.PHONY: ui api dev clean health reindex generate-dry

API_HOST ?= 127.0.0.1
API_PORT ?= 8000
SMOKE_THEME ?= finance

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

## Быстрый smoke-тест API (health)
health:
        @echo "GET /api/health"
        @curl -sS -w '\nHTTP %{http_code}\n' http://$(API_HOST):$(API_PORT)/api/health || true

## Быстрый smoke-тест API (reindex)
reindex:
        @echo "POST /api/reindex (theme=$(SMOKE_THEME))"
        @curl -sS -w '\nHTTP %{http_code}\n' \
                -H 'Content-Type: application/json' \
                -d "{\"theme\":\"$(SMOKE_THEME)\"}" \
                http://$(API_HOST):$(API_PORT)/api/reindex || true

## Быстрый smoke-тест API (generate dry-run)
generate-dry:
        @echo "POST /api/generate (dry_run, theme=$(SMOKE_THEME))"
        @curl -sS -w '\nHTTP %{http_code}\n' \
                -H 'Content-Type: application/json' \
                -d "{\"theme\":\"$(SMOKE_THEME)\",\"dry_run\":true,\"k\":0,\"data\":{\"theme\":\"Smoke test\",\"goal\":\"demo\"}}" \
                http://$(API_HOST):$(API_PORT)/api/generate || true
