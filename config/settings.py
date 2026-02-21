ENV_CONFIG = {
    "initial_cash": 100_000.0,
    "max_inventory": 100,
    "tick_size": 0.01,
    "lot_size": 10,
    "max_spread": 0.20,
    "min_spread": 0.01,
    "episode_length": 1000,
    "market_impact": 0.001,
    "adverse_selection_prob": 0.15,
    "inventory_penalty": 0.001,
    "drift": 0.0,
    "volatility": 0.02,
    "mean_revert_speed": 0.05,
    "seed": 42,
}
TRAINING_CONFIG = {
    "total_timesteps": 500_000,
    "learning_rate": 3e-4,
    "batch_size": 256,
    "buffer_size": 100_000,
    "learning_starts": 10_000,
    "tau": 0.005,
    "gamma": 0.99,
    "ent_coef": "auto",
    "train_freq": 1,
    "gradient_steps": 1,
    "eval_freq": 10_000,
    "eval_episodes": 20,
    "n_envs": 1,
    "log_interval": 100,
}
PATHS = {
    "model_dir": "models",
    "best_model": "models/best_model",
    "final_model": "models/final_model",
    "tensorboard_log": "models/tb_logs",
    "training_plots": "models/training_plots",
}
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "redis_host": "localhost",
    "redis_port": 6379,
    "redis_db": 0,
    "agent_state_key": "agent:state",
    "quotes_key": "agent:quotes",
}
MONITORING_CONFIG = {
    "prometheus_port": 8001,
}
