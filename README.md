# ReflexDesk

Мультиагентный инвестиционный research-деск: собирает данные (рынок, SEC, новости, FRED), проводит анализ с дебатами Bull/Bear, проверяет числа валидатором, выпускает отчёт, записывает прогнозы и по наступлении горизонта сверяет их с рынком, извлекая уроки в память.

## Требования

- Docker Desktop (WSL2, ≥ 12 ГБ RAM для VM), NVIDIA GPU 8 ГБ VRAM, драйвер с CUDA ≥ 12.8
- Ключи: `DEEPSEEK_API_KEY` (ориг ключ), `QWEN_API_KEY` (использую ключ провайдера polza.ai, с оригинальным ключом квена не совместимо), `SEC_USER_AGENT`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_CHAT_IDS`, `TELEGRAM_CHAT_ID`, `FRED_API_KEY` (ключ бесплатный, получить можно тут: https://fred.stlouisfed.org/)

## Как пользоваться

Основной интерфейс - Telegram-бот, нужно предварительно его создать и выпустить токен. Доступ только у чатов из `TELEGRAM_ALLOWED_CHAT_IDS`.

- каждое сообщение без команды - запрос на анализ: `Проанализируй NVDA на 90 дней`, `Проанализируй MSFT 2024-03-01`, можно и без дат и конкретного тикера: `Стоит ли сейчас брать биткоин?`;
- `/runs`, `/status <id>`, `/report <id>`, `/predictions [ТИКЕР]`, `/metrics`, `/help`;
- готовый отчёт и сверенные прогнозы бот присылает сам.

Свой chat_id: напишите боту что угодно — если чата нет в списке, бот ответит его id с просьбой передать администратору для доступа.

## Запуск

```bash
cp .env.example .env          
docker compose up -d --build  
curl localhost:8000/readyz    
```

Курлы руками:

```bash
curl -X POST localhost:8000/v1/analyze -H 'Content-Type: application/json' -d '{"query":"Analyze NVDA"}'
curl localhost:8000/v1/runs/<run_id>          # статус и выжимка
curl localhost:8000/v1/runs/<run_id>/report   # Markdown + ссылка на PDF
```


## Observability + evals

| Что | Адрес | Вход |
|---|---|---|
| API (Swagger) | http://localhost:8000/docs | — |
| Трейсы, траектории, golden-set | Langfuse http://localhost:3000 | `admin@reflexdesk.local` / `LANGFUSE_ADMIN_PASSWORD` |
| Дашборд, логи по `run_id` | Grafana http://localhost:3001 | `admin` / `GRAFANA_PASSWORD` |
| Метрики и алерты | Prometheus http://localhost:9090 (`/alerts`) | — |
| Сводка hit rate / Brier / cost | http://localhost:8000/v1/metrics/summary | — |
| Алерты | Telegram-бот из `TELEGRAM_BOT_TOKEN` | — |


C4 диаграммы + sequence d: `docs/c4/`.
