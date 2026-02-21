import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import ENV_CONFIG

class LimitOrderBookEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        cfg = config or ENV_CONFIG
        self.initial_cash = cfg["initial_cash"]
        self.max_inventory = cfg["max_inventory"]
        self.lot_size = cfg["lot_size"]
        self.max_spread = cfg["max_spread"]
        self.min_spread = cfg["min_spread"]
        self.episode_length = cfg["episode_length"]
        self.market_impact = cfg["market_impact"]
        self.adverse_selection_prob = cfg["adverse_selection_prob"]
        self.inventory_penalty = cfg["inventory_penalty"]
        self.drift = cfg["drift"]
        self.volatility = cfg["volatility"]
        self.mean_revert_speed = cfg["mean_revert_speed"]
        self.render_mode = render_mode
        self.initial_mid_price = 100.0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng(cfg["seed"])

    def _map_spread(self, v):
        return self.min_spread + ((v + 1.0) / 2.0) * (self.max_spread - self.min_spread)

    def _fill_prob(self, hs):
        return 0.7 * np.exp(-20.0 * hs)

    def _next_price(self):
        noise = self.rng.normal(0, self.volatility)
        mr = -self.mean_revert_speed * (self.mid_price - self.initial_mid_price)
        return max(self.mid_price + self.drift + mr + noise, 1.0)

    def _ofi(self):
        if len(self.price_hist) < 2:
            return 0.0
        return float(np.clip(np.sign(np.diff(self.price_hist[-10:])).mean(), -1, 1))

    def _obs(self, bf=0, af=0, pm=0.0, sp=0.0):
        upnl = self.inventory * self.lot_size * (self.mid_price - self.initial_mid_price)
        vol = float(np.std(np.diff(self.price_hist[-20:]))) if len(self.price_hist) > 20 else self.volatility
        return np.array([
            (self.mid_price - self.initial_mid_price) / self.initial_mid_price,
            sp / self.max_spread,
            self.inventory / self.max_inventory,
            self.cash / self.initial_cash,
            upnl / self.initial_cash,
            self.realized_pnl / self.initial_cash,
            vol / self.volatility,
            self._ofi(),
            1.0 - (self.step_count / self.episode_length),
            float(bf), float(af),
            pm / (self.volatility + 1e-8),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.mid_price = self.initial_mid_price
        self.inventory = 0
        self.cash = self.initial_cash
        self.realized_pnl = 0.0
        self.step_count = 0
        self.price_hist = [self.mid_price]
        return self._obs(), {}

    def step(self, action):
        bhs = self._map_spread(float(action[0]))
        ahs = self._map_spread(float(action[1]))
        bid = round(self.mid_price - bhs, 4)
        ask = round(self.mid_price + ahs, 4)
        adverse = self.rng.random() < self.adverse_selection_prob
        bf = int(self.rng.random() < self._fill_prob(bhs))
        af = int(self.rng.random() < self._fill_prob(ahs))
        if self.inventory >= self.max_inventory: bf = 0
        if self.inventory <= -self.max_inventory: af = 0

        step_pnl, adv_cost = 0.0, 0.0
        if bf:
            self.cash -= bid * self.lot_size + self.market_impact * self.lot_size * bid
            self.inventory += 1
            if adverse: adv_cost += bhs * self.lot_size * 0.5
        if af:
            self.cash += ask * self.lot_size - self.market_impact * self.lot_size * ask
            self.inventory -= 1
            if adverse: adv_cost += ahs * self.lot_size * 0.5
        if bf and af:
            step_pnl += (ask - bid) * self.lot_size

        old_price = self.mid_price
        self.mid_price = self._next_price()
        pm = self.mid_price - old_price
        self.price_hist.append(self.mid_price)
        mtm = self.inventory * self.lot_size * pm
        self.realized_pnl += step_pnl
        self.step_count += 1

        reward = step_pnl + mtm - self.inventory_penalty * abs(self.inventory) - adv_cost
        done = self.step_count >= self.episode_length
        if done and self.inventory != 0:
            reward -= abs(self.inventory) * self.lot_size * self.mid_price * 0.005

        info = {
            "mid_price": self.mid_price, "bid_price": bid, "ask_price": ask,
            "bid_half_spread": bhs, "ask_half_spread": ahs,
            "bid_fill": bf, "ask_fill": af,
            "inventory": self.inventory, "cash": self.cash,
            "realized_pnl": self.realized_pnl, "step": self.step_count,
        }
        return self._obs(bf, af, pm, bhs + ahs), float(reward), done, False, info

    def render(self):
        if self.render_mode == "human":
            pnl = self.realized_pnl + self.inventory * self.lot_size * (self.mid_price - self.initial_mid_price)
            print(f"Step {self.step_count:4d} | Mid: {self.mid_price:8.4f} | Inv: {self.inventory:+4d} | PnL: {pnl:+8.2f}")

if __name__ == "__main__":
    env = LimitOrderBookEnv()
    obs, _ = env.reset()
    print(f"✓ Obs shape: {obs.shape} | Action space: {env.action_space}")
    total = 0
    for i in range(200):
        obs, r, done, _, info = env.step(env.action_space.sample())
        total += r
        if i % 50 == 0: env.render()
        if done: break
    print(f"✓ Sanity check passed. Reward: {total:.2f}, Final inv: {info['inventory']}")
