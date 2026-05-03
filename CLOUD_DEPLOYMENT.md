# Cloud deployment plan

This project now has a cloud-style one-tick runner:

- `coin_mvp.cloud_tick`
- Vercel API route: `api/tick.py`
- Vercel cron config: `vercel.json`

## How it works

Vercel Cron calls `/api/tick` every 5 minutes.

Each call:

1. Loads the latest paper-trading state.
2. Fetches public Upbit market data.
3. Runs exactly one strategy tick.
4. Saves updated cash, positions, risk state, trades, and events.
5. Refreshes a temporary dashboard HTML file for smoke checks.

This is the correct shape for Vercel because Vercel does not run a permanently alive bot process.

The live dashboard is served from `/api/report`. It reads the latest persisted state/trades/events and renders the same console UI on demand.

## Persistent storage

For local smoke tests, the runner can still use local files. For Vercel production, set a persistent Redis/KV store so each 5-minute function call can continue from the previous seed, positions, trades, and risk state.

Supported REST-compatible options:

- Upstash Redis
- Vercel KV / Redis-compatible REST envs

Storage keys:

- `<prefix>:state`
- `<prefix>:trades`
- `<prefix>:events`

Default prefix is `coin_mvp`.

## Local smoke test

```powershell
python -m coin_mvp.cloud_tick --config config.cloud.json --ticks 1 --top-markets 12 --request-delay 0.35 --output reports/cloud_tick_smoke.html
```

## Vercel environment variables

Recommended:

- `COIN_MVP_CONFIG=config.cloud.json`
- `TOP_MARKETS=30`
- `REQUEST_DELAY=0.35`
- `REPORT_OUTPUT=/tmp/coin_mvp/report.html`
- `CRON_SECRET=<random secret>`
- `COIN_MVP_STORAGE=upstash`
- `UPSTASH_REDIS_REST_URL=<your Upstash REST URL>`
- `UPSTASH_REDIS_REST_TOKEN=<your Upstash REST token>`
- `COIN_MVP_STORAGE_PREFIX=coin_mvp`

Vercel KV-compatible alternatives:

- `KV_REST_API_URL=<your KV REST URL>`
- `KV_REST_API_TOKEN=<your KV REST token>`

If `CRON_SECRET` is set, callers must provide either:

- `x-cron-secret: <secret>`
- `Authorization: Bearer <secret>`

## Production URLs

- Cron/manual tick: `/api/tick`
- Dashboard: `/api/report`

Keep this as paper trading until order execution, exchange keys, kill-switches, and private API signing are reviewed separately.
