import os, sys, argparse, json, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from env.lob_env import LimitOrderBookEnv
from config.settings import ENV_CONFIG, TRAINING_CONFIG, PATHS

class MetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_pnls, self.ep_rewards = [], []
        self._ep_reward = 0.0
    def _on_step(self):
        self._ep_reward += self.locals.get("rewards", [0])[0]
        if self.locals.get("dones", [False])[0]:
            info = (self.locals.get("infos") or [{}])[0]
            self.ep_pnls.append(info.get("realized_pnl", 0.0))
            self.ep_rewards.append(self._ep_reward)
            self._ep_reward = 0.0
            if len(self.ep_pnls) % 10 == 0:
                recent = self.ep_pnls[-50:]
                sharpe = np.mean(recent) / (np.std(recent) + 1e-8) * np.sqrt(252)
                self.logger.record("custom/mean_pnl_50ep", np.mean(recent))
                self.logger.record("custom/sharpe_50ep", sharpe)
        return True

def make_env(seed=0):
    def _init():
        env = LimitOrderBookEnv(config=ENV_CONFIG)
        return Monitor(env)
    return _init

def train(args):
    for p in [PATHS["model_dir"], PATHS["training_plots"], PATHS["tensorboard_log"]]:
        os.makedirs(p, exist_ok=True)
    print(f"\n{'='*50}\n  SAC MARKET MAKER TRAINING\n  Timesteps: {args.timesteps:,} | Seed: {args.seed}\n{'='*50}\n")
    train_env = make_vec_env(make_env(args.seed), n_envs=1, seed=args.seed)
    eval_env  = make_vec_env(make_env(args.seed+1), n_envs=1, seed=args.seed+1)
    logger = configure(PATHS["tensorboard_log"], ["stdout", "tensorboard"])
    model = SAC(
        "MlpPolicy", train_env,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        batch_size=TRAINING_CONFIG["batch_size"],
        buffer_size=TRAINING_CONFIG["buffer_size"],
        learning_starts=TRAINING_CONFIG["learning_starts"],
        tau=TRAINING_CONFIG["tau"], gamma=TRAINING_CONFIG["gamma"],
        ent_coef=TRAINING_CONFIG["ent_coef"],
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        verbose=1, seed=args.seed,
        tensorboard_log=PATHS["tensorboard_log"], device="auto",
    )
    model.set_logger(logger)
    metrics_cb = MetricsCallback()
    callbacks = [
        EvalCallback(eval_env, best_model_save_path=PATHS["model_dir"],
                     eval_freq=TRAINING_CONFIG["eval_freq"],
                     n_eval_episodes=TRAINING_CONFIG["eval_episodes"],
                     deterministic=True, verbose=1),
        CheckpointCallback(save_freq=50_000, save_path=PATHS["model_dir"],
                           name_prefix="sac_ckpt", verbose=1),
        metrics_cb,
    ]
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                log_interval=TRAINING_CONFIG["log_interval"], tb_log_name="SAC_LOB")
    elapsed = time.time() - t0
    model.save(PATHS["final_model"])
    meta = {
        "timesteps": args.timesteps, "seed": args.seed,
        "elapsed_seconds": round(elapsed, 1),
        "total_episodes": len(metrics_cb.ep_pnls),
        "mean_pnl": round(float(np.mean(metrics_cb.ep_pnls)), 2) if metrics_cb.ep_pnls else 0,
        "sharpe": round(float(np.mean(metrics_cb.ep_pnls) / (np.std(metrics_cb.ep_pnls) + 1e-8) * np.sqrt(252)), 3) if len(metrics_cb.ep_pnls) > 1 else 0,
    }
    with open(f"{PATHS['model_dir']}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    # Plot
    if metrics_cb.ep_pnls:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("SAC Training Curves")
        axes[0].plot(metrics_cb.ep_pnls, alpha=0.4, color="steelblue")
        w = max(1, len(metrics_cb.ep_pnls)//20)
        axes[0].plot(np.convolve(metrics_cb.ep_pnls, np.ones(w)/w, "valid"), color="steelblue")
        axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)
        axes[0].set_title("Episode PnL"); axes[0].set_xlabel("Episode")
        axes[1].plot(np.cumsum(metrics_cb.ep_pnls), color="green")
        axes[1].set_title("Cumulative PnL"); axes[1].set_xlabel("Episode")
        plt.tight_layout()
        plt.savefig(f"{PATHS['training_plots']}/curves.png", dpi=120)
        plt.close()
    print(f"\n{'='*50}\n  DONE in {elapsed/60:.1f} min\n  Episodes: {meta['total_episodes']} | Mean PnL: {meta['mean_pnl']} | Sharpe: {meta['sharpe']}\n  Best model: {PATHS['model_dir']}/best_model.zip\n{'='*50}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=TRAINING_CONFIG["total_timesteps"])
    p.add_argument("--seed", type=int, default=42)
    train(p.parse_args())
