"""
DQN Training Script for Circuit Breaker RL
Compare DQN performance against PPO baseline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from circuit_breaker_env import CircuitBreakerEnv


class RewardCallback(BaseCallback):
    """Callback to track episode rewards"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = []

    def _on_step(self):
        self.current_rewards.append(self.locals['rewards'][0])

        if self.locals['dones'][0]:
            episode_reward = sum(self.current_rewards)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(len(self.current_rewards))
            self.current_rewards = []

            if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                print(f"Episode {len(self.episode_rewards)}: Reward = {episode_reward:.2f}")

        return True


def train_dqn():
    """Train DQN agent for comparison with PPO"""

    print("=" * 70)
    print("DQN Training - Circuit Breaker RL")
    print("=" * 70)

    # Create training environment
    env = CircuitBreakerEnv(data_path='data/behavioral_dataset.csv')
    env = Monitor(env)

    # Create evaluation environment
    eval_env = CircuitBreakerEnv(data_path='data/behavioral_dataset.csv')
    eval_env = Monitor(eval_env)

    # Create reward tracking callback
    reward_callback = RewardCallback(verbose=1)

    # Create evaluation callback
    os.makedirs("./logs/dqn_best_model", exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/dqn_best_model",
        log_path="./logs/dqn_eval/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )

    # Initialize DQN
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./dqn_circuit_breaker_tensorboard/"
    )

    print("\nStarting DQN training:")
    print(f"   - Total timesteps: 500,000")
    print(f"   - Buffer size: 100,000")
    print(f"   - Batch size: 64")
    print(f"   - Learning rate: 1e-4")
    print(f"   - Exploration: eps 1.0 -> 0.05 (20% of training)")
    print(f"   - Target update interval: 1000 steps")
    print(f"   - Observation: 7 features")
    print(f"   - Actions: 5 discrete\n")

    # Train 500k timesteps
    model.learn(
        total_timesteps=500000,
        callback=[reward_callback, eval_callback],
        progress_bar=True
    )

    # Save final model
    model.save("dqn_circuit_breaker")
    print("\nModel saved: dqn_circuit_breaker.zip")
    print("Best model saved: ./logs/dqn_best_model/best_model.zip")

    return model, reward_callback


def plot_reward_curve(callback):
    """Plot DQN training reward curve"""

    rewards = callback.episode_rewards

    if len(rewards) == 0:
        print("No episodes completed!")
        return

    window = min(50, max(1, len(rewards) // 10))
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Raw rewards
    ax1.plot(rewards, alpha=0.3, color='darkorange', label='Episode Reward')
    ax1.plot(range(window-1, len(rewards)), moving_avg,
             color='red', linewidth=2, label=f'Moving Avg (window={window})')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('DQN Training - Episode Rewards', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reward distribution
    ax2.hist(rewards, bins=50, color='lightsalmon', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(rewards), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    ax2.axvline(np.median(rewards), color='green', linestyle='--',
                linewidth=2, label=f'Median: {np.median(rewards):.2f}')
    ax2.set_xlabel('Episode Reward', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Reward Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('DQN Circuit Breaker Training', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dqn_reward_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 70)
    print("DQN TRAINING STATISTICS")
    print("=" * 70)
    print(f"Total Episodes:  {len(rewards)}")
    print(f"Mean Reward:     {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Min Reward:      {np.min(rewards):.2f}")
    print(f"Max Reward:      {np.max(rewards):.2f}")
    print(f"Median Reward:   {np.median(rewards):.2f}")
    print("=" * 70)
    print("Plot saved: dqn_reward_curve.png\n")


def compare_ppo_dqn(dqn_callback):
    """Compare PPO vs DQN reward curves if PPO model exists"""

    from stable_baselines3 import PPO

    try:
        ppo_model = PPO.load("ppo_circuit_breaker")
    except:
        print("PPO model not found, skipping comparison plot.")
        return

    env = CircuitBreakerEnv(data_path='data/behavioral_dataset.csv')

    # Evaluate PPO
    ppo_rewards = []
    for _ in range(50):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        ppo_rewards.append(episode_reward)

    # Evaluate DQN (from training callback)
    dqn_rewards = dqn_callback.episode_rewards

    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    # DQN training curve
    dqn_window = min(50, max(1, len(dqn_rewards) // 10))
    dqn_moving_avg = np.convolve(dqn_rewards, np.ones(dqn_window)/dqn_window, mode='valid')
    ax.plot(range(dqn_window-1, len(dqn_rewards)), dqn_moving_avg,
            color='red', linewidth=2, label=f'DQN (Mean: {np.mean(dqn_rewards[-100:]):.2f})')

    # PPO evaluation line
    ax.axhline(np.mean(ppo_rewards), color='blue', linestyle='--',
               linewidth=2, label=f'PPO Eval (Mean: {np.mean(ppo_rewards):.2f})')
    ax.axhspan(np.mean(ppo_rewards) - np.std(ppo_rewards),
               np.mean(ppo_rewards) + np.std(ppo_rewards),
               alpha=0.1, color='blue')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('PPO vs DQN Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ppo_vs_dqn_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 70)
    print("PPO vs DQN COMPARISON")
    print("=" * 70)
    print(f"PPO Mean Reward (eval):      {np.mean(ppo_rewards):.2f} +/- {np.std(ppo_rewards):.2f}")
    print(f"DQN Mean Reward (last 100):  {np.mean(dqn_rewards[-100:]):.2f} +/- {np.std(dqn_rewards[-100:]):.2f}")
    print(f"Winner: {'PPO' if np.mean(ppo_rewards) > np.mean(dqn_rewards[-100:]) else 'DQN'}")
    print("=" * 70)
    print("Plot saved: ppo_vs_dqn_comparison.png\n")


def evaluate_model(model, env, n_episodes=10):
    """Evaluate trained DQN model"""

    print("\n" + "=" * 70)
    print(f"EVALUATING DQN MODEL - {n_episodes} episodes")
    print("=" * 70)

    episode_rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        print(f"Episode {ep+1:2d}: Reward = {episode_reward:7.2f}, Steps = {steps:3d}")

    print("=" * 70)
    print(f"Mean Reward:  {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print("=" * 70)


if __name__ == "__main__":
    # Train DQN
    model, callback = train_dqn()

    # Plot reward curve
    plot_reward_curve(callback)

    # Evaluate final model
    env = CircuitBreakerEnv(data_path='data/behavioral_dataset.csv')
    evaluate_model(model, env, n_episodes=10)

    # Load and evaluate best model
    print("\n" + "=" * 70)
    print("EVALUATING BEST DQN MODEL (from evaluation callback)")
    print("=" * 70)
    try:
        best_model = DQN.load("./logs/dqn_best_model/best_model")
        evaluate_model(best_model, env, n_episodes=10)
    except:
        print("Best DQN model not found, skipping")

    # Compare PPO vs DQN
    compare_ppo_dqn(callback)

    print("\nDQN Training complete! Files saved:")
    print("   - dqn_circuit_breaker.zip (final model)")
    print("   - ./logs/dqn_best_model/best_model.zip (best model)")
    print("   - dqn_reward_curve.png (training plot)")
    print("   - ppo_vs_dqn_comparison.png (comparison plot)")
    print("   - dqn_circuit_breaker_tensorboard/ (logs)\n")
