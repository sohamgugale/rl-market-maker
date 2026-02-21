import os, sys, time, json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import SAC
from env.lob_env import LimitOrderBookEnv
from config.settings import ENV_CONFIG, PATHS, API_CONFIG

class InferenceEngine:
    def __init__(self, model_path=None, use_redis=True):
        path = model_path or f"{PATHS['best_model']}.zip"
        if not os.path.exists(path):
            path = f"{PATHS['final_model']}.zip"
        print(f"Loading model: {path}")
        self.model = SAC.load(path)
        self.env = LimitOrderBookEnv(config=ENV_CONFIG)
        self.obs, _ = self.env.reset()
        self.state = {}
        self.step_count = 0
        self.episode_count = 0
        self.redis = None
        if use_redis:
            try:
                import redis as redis_lib
                r = redis_lib.Redis(host=API_CONFIG["redis_host"], port=API_CONFIG["redis_port"],
                                    db=API_CONFIG["redis_db"], decode_responses=True,
                                    socket_connect_timeout=2)
                r.ping()
                self.redis = r
                print("Redis connected.")
            except Exception as e:
                print(f"Redis unavailable ({e}). Continuing without it.")

    def step(self):
        action, _ = self.model.predict(self.obs, deterministic=True)
        obs_next, reward, done, _, info = self.env.step(action)
        self.state = {
            "step": info["step"], "episode": self.episode_count,
            "mid_price": round(info["mid_price"], 4),
            "bid_price": round(info["bid_price"], 4),
            "ask_price": round(info["ask_price"], 4),
            "bid_half_spread": round(info["bid_half_spread"], 4),
            "ask_half_spread": round(info["ask_half_spread"], 4),
            "bid_fill": int(info["bid_fill"]), "ask_fill": int(info["ask_fill"]),
            "inventory": int(info["inventory"]), "cash": round(info["cash"], 2),
            "realized_pnl": round(info["realized_pnl"], 2),
            "reward": round(float(reward), 4),
            "timestamp": time.time(),
        }
        if self.redis:
            try:
                self.redis.setex(API_CONFIG["agent_state_key"], 60, json.dumps(self.state))
                self.redis.setex(API_CONFIG["quotes_key"], 10, json.dumps({
                    k: self.state[k] for k in ["bid_price","ask_price","mid_price","bid_half_spread","ask_half_spread","timestamp"]
                }))
            except Exception: pass
        if done:
            print(f"[Ep {self.episode_count}] Steps: {info['step']} | PnL: {info['realized_pnl']:.2f} | Inv: {info['inventory']}")
            self.obs, _ = self.env.reset()
            self.episode_count += 1
        else:
            self.obs = obs_next
        self.step_count += 1
        return self.state

    def run(self, interval=0.05, max_steps=None):
        print(f"Inference running (interval={interval}s). Ctrl+C to stop.\n")
        i = 0
        try:
            while True:
                if max_steps and i >= max_steps: break
                s = self.step()
                if i % 100 == 0:
                    print(f"[{i:5d}] Ep={s['episode']} Mid={s['mid_price']:.4f} "
                          f"Bid={s['bid_price']:.4f} Ask={s['ask_price']:.4f} "
                          f"Inv={s['inventory']:+3d} PnL={s['realized_pnl']:+.2f}")
                time.sleep(interval)
                i += 1
        except KeyboardInterrupt:
            print(f"\nStopped. {i} steps, {self.episode_count} episodes.")

if __name__ == "__main__":
    engine = InferenceEngine(use_redis=False)
    print("Running 1 episode...")
    while True:
        s = engine.step()
        if s["step"] == 0: break
    print(f"Done. PnL: {s['realized_pnl']:.2f}, Inventory: {s['inventory']}")
