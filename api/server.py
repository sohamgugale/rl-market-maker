import os, sys, time, threading
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.engine import InferenceEngine
from config.settings import API_CONFIG

engine: Optional[InferenceEngine] = None
_paused = threading.Event(); _paused.set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    try:
        engine = InferenceEngine(use_redis=True)
        def _loop():
            while True:
                if _paused.is_set(): engine.step()
                time.sleep(0.1)
        threading.Thread(target=_loop, daemon=True).start()
        print("Inference engine started.")
    except Exception as e:
        print(f"Could not start engine: {e}. Train a model first.")
    yield

app = FastAPI(title="RL Market Maker", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_start = time.time()

@app.get("/health")
def health():
    return {"status": "ok", "uptime": round(time.time()-_start, 1), "engine": engine is not None}

@app.get("/quotes")
def quotes():
    if not engine: raise HTTPException(503, "Engine not ready. Train a model first.")
    s = engine.state
    if not s: raise HTTPException(503, "No quotes yet — give it a moment.")
    return {"bid_price": s["bid_price"], "ask_price": s["ask_price"],
            "mid_price": s["mid_price"], "spread": round(s["bid_half_spread"]+s["ask_half_spread"],4),
            "timestamp": s["timestamp"]}

@app.get("/state")
def state():
    if not engine: raise HTTPException(503, "Engine not ready.")
    if not engine.state: raise HTTPException(503, "No state yet.")
    return engine.state

@app.get("/metrics")
def metrics():
    if not engine: raise HTTPException(503, "Engine not ready.")
    s = engine.state
    return {"total_steps": engine.step_count, "total_episodes": engine.episode_count,
            "inventory": s.get("inventory",0), "realized_pnl": s.get("realized_pnl",0),
            "uptime": round(time.time()-_start,1)}

@app.post("/control/pause")
def pause(): _paused.clear(); return {"status": "paused"}

@app.post("/control/resume")
def resume(): _paused.set(); return {"status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=API_CONFIG["host"], port=API_CONFIG["port"])
