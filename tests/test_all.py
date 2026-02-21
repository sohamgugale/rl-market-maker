import sys, os, time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.lob_env import LimitOrderBookEnv
from config.settings import ENV_CONFIG

def run(name, fn):
    try: fn(); print(f"  ✓ {name}")
    except Exception as e: print(f"  ✗ {name}: {e}")

def test_env():
    env = LimitOrderBookEnv()
    obs, _ = env.reset()
    assert obs.shape == (12,)
    assert env.action_space.shape == (2,)
    # Full episode
    done = False
    total_r = 0
    while not done:
        obs, r, done, _, info = env.step(env.action_space.sample())
        total_r += r
        assert np.all(np.isfinite(obs)), "Non-finite obs"
        assert np.isfinite(r), "Non-finite reward"
        assert abs(info["inventory"]) <= ENV_CONFIG["max_inventory"]
    assert info["step"] == ENV_CONFIG["episode_length"]
    # Throughput
    env.reset()
    t0 = time.time()
    for _ in range(1000):
        _, _, done, _, _ = env.step(env.action_space.sample())
        if done: env.reset()
    sps = 1000 / (time.time() - t0)
    assert sps > 500, f"Too slow: {sps:.0f} steps/sec"
    print(f"    ({sps:.0f} steps/sec)")

def test_redis():
    import redis
    r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=2)
    r.ping()
    r.set("test:mm", "hello", ex=5)
    assert r.get("test:mm") == b"hello"
    r.delete("test:mm")

def test_api():
    import requests
    h = requests.get("http://localhost:8000/health", timeout=2).json()
    assert h["status"] == "ok"
    requests.get("http://localhost:8000/quotes", timeout=2)
    requests.get("http://localhost:8000/state", timeout=2)

print("\n" + "="*40)
print("  RL MARKET MAKER — TESTS")
print("="*40)
run("Environment (full episode, obs/reward finite, throughput)", test_env)
run("Redis (connect, write, read)", test_redis)
run("API (health, quotes, state)", test_api)
print("="*40)
