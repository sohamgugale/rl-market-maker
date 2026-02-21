# RL Market Maker

A production-grade **Reinforcement Learning market-making agent** trained with Soft Actor-Critic (SAC) on a simulated Limit Order Book (LOB). The agent learns optimal bid/ask spread quoting under realistic market microstructure constraints — adverse selection, inventory risk, and market impact.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Redis](https://img.shields.io/badge/Redis-7-red)

## What It Does

Most ML projects predict prices. This one **makes markets** — continuously quoting bid and ask prices to earn the spread while managing inventory risk, exactly like a prop trading desk or HFT firm.

The SAC agent learns to:
- Quote tight spreads when order flow is uninformed (earn more)
- Widen spreads when adverse selection risk is high (protect PnL)
- Keep inventory near zero to avoid directional exposure
- Adapt quoting in real time based on volatility and market state

## Results

| Metric | Value |
|--------|-------|
| Training timesteps | 500,000 |
| Mean episode PnL | **$35.34** |
| Sharpe ratio | **59.7** |
| Environment throughput | **30,000+ steps/sec** |
| API response latency | <100ms |

## Architecture
```
LOB Simulator (Gymnasium)
        │
        ▼
  SAC Agent (PyTorch)
        │
        ▼
 Inference Engine
        │
   ┌────┴────┐
   ▼         ▼
 Redis    Prometheus
   │
   ▼
FastAPI (REST API)
```

## Quickstart
```bash
git clone https://github.com/sohamgugale/rl-market-maker.git
cd rl-market-maker
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python training/train.py --timesteps 50000
redis-server --daemonize yes
uvicorn api.server:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/quotes
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status |
| `/quotes` | GET | Live bid/ask prices |
| `/state` | GET | Full agent state |
| `/metrics` | GET | Performance metrics |
| `/control/pause` | POST | Pause inference |
| `/control/resume` | POST | Resume inference |

## Docker
```bash
docker compose up -d
curl http://localhost:8000/health
```

## Skills Demonstrated

- Reinforcement learning (SAC, custom reward shaping, Gymnasium)
- Market microstructure (adverse selection, inventory risk, LOB dynamics)
- Production ML engineering (inference engine, model serving, monitoring)
- Backend systems (FastAPI, Redis, Prometheus, Docker)

## License

MIT
