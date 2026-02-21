import time
from prometheus_client import Gauge, Counter, Histogram, start_http_server

MM_PNL       = Gauge("mm_pnl_total", "Realized PnL")
MM_INVENTORY = Gauge("mm_inventory", "Current inventory")
MM_CASH      = Gauge("mm_cash", "Cash balance")
MM_BID_FILLS = Counter("mm_bid_fills_total", "Bid fills")
MM_ASK_FILLS = Counter("mm_ask_fills_total", "Ask fills")
MM_STEPS     = Counter("mm_steps_total", "Total steps")
MM_EPISODES  = Counter("mm_episodes_total", "Episodes completed")
MM_SPREAD    = Histogram("mm_spread", "Total quoted spread",
                         buckets=[0.02,0.04,0.06,0.10,0.15,0.20,0.30,0.40])
MM_LATENCY   = Histogram("mm_step_latency_seconds", "Step latency",
                         buckets=[0.001,0.005,0.01,0.025,0.05,0.1])

class MetricsCollector:
    def __init__(self): self._last_ep = 0
    def update(self, state: dict, latency: float = 0.0):
        if not state: return
        MM_PNL.set(state.get("realized_pnl", 0))
        MM_INVENTORY.set(state.get("inventory", 0))
        MM_CASH.set(state.get("cash", 0))
        if state.get("bid_fill"): MM_BID_FILLS.inc()
        if state.get("ask_fill"): MM_ASK_FILLS.inc()
        MM_STEPS.inc()
        spread = state.get("bid_half_spread",0) + state.get("ask_half_spread",0)
        MM_SPREAD.observe(spread)
        MM_LATENCY.observe(latency)
        ep = state.get("episode", 0)
        if ep > self._last_ep:
            MM_EPISODES.inc(ep - self._last_ep)
            self._last_ep = ep

def start_metrics_server(port=8001):
    start_http_server(port)
    print(f"Prometheus metrics → http://localhost:{port}/metrics")
