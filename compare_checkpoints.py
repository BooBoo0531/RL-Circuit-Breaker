"""
Compare best checkpoint vs final model for PPO and DQN.
Run from project root after training:
    python compare_checkpoints.py
"""

import sys
import numpy as np
sys.path.insert(0, 'src')

from stable_baselines3 import PPO, DQN
from circuit_breaker_env import CircuitBreakerEnv

N_EVAL = 30
DATA_PATH = 'data/behavioral_dataset.csv'

MODELS = [
    {
        'name': 'PPO',
        'algo': PPO,
        'best': 'logs/ppo_best_model/best_model',
        'final': 'ppo_circuit_breaker',
    },
    {
        'name': 'DQN',
        'algo': DQN,
        'best': 'logs/dqn_best_model/best_model',
        'final': 'dqn_circuit_breaker',
    },
]


def evaluate(model, env, n=N_EVAL):
    rewards = []
    for _ in range(n):
        obs, _ = env.reset()
        ep_r, done = 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    return np.mean(rewards), np.std(rewards)


def main():
    env = CircuitBreakerEnv(data_path=DATA_PATH)

    print("=" * 60)
    print("  BEST CHECKPOINT vs FINAL MODEL COMPARISON")
    print(f"  Episodes per evaluation: {N_EVAL}")
    print("=" * 60)

    for m in MODELS:
        print(f"\n--- {m['name']} ---")

        # Load best checkpoint
        try:
            best_model = m['algo'].load(m['best'])
            bm, bs = evaluate(best_model, env)
            best_str = f"{bm:.2f} +/- {bs:.2f}"
        except Exception as e:
            bm, best_str = None, f"not found ({e})"

        # Load final model
        try:
            final_model = m['algo'].load(m['final'])
            fm, fs = evaluate(final_model, env)
            final_str = f"{fm:.2f} +/- {fs:.2f}"
        except Exception as e:
            fm, final_str = None, f"not found ({e})"

        print(f"  Best checkpoint : {best_str}")
        print(f"  Final model     : {final_str}")

        if bm is not None and fm is not None:
            if bm > fm * 1.05:
                verdict = "BEST wins → policy degradation occurred, use best checkpoint"
            elif fm > bm * 1.05:
                verdict = "FINAL wins → EvalCallback may have issues, check eval_freq"
            else:
                verdict = "SIMILAR → no significant degradation, either model is fine"
            print(f"  Verdict         : {verdict}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
