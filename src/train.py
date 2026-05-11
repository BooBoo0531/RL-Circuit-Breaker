"""
Simplified PPO Training Script for Circuit Breaker RL
Reset to simplest working config with clear reward signal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from circuit_breaker_env import CircuitBreakerEnv


# ============================================================================
# REWARD TRACKING CALLBACK
# ============================================================================

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


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_ppo():
    """Train PPO agent with simplified, working configuration"""
    
    print("=" * 70)
    print("PPO Training - SIMPLIFIED CONFIG with CLEAR REWARD SIGNAL")
    print("=" * 70)
    
    # Create training environment
    env = CircuitBreakerEnv(data_path='data/behavioral_dataset.csv')
    env = Monitor(env)
    
    # Create evaluation environment
    eval_env = CircuitBreakerEnv(data_path='data/behavioral_dataset.csv')
    eval_env = Monitor(eval_env)
    
    # Create reward tracking callback
    reward_callback = RewardCallback(verbose=1)
    
    # Create evaluation callback (evaluates every 10k steps, saves best model)
    os.makedirs("./logs/", exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Initialize PPO with SIMPLIFIED working config
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,      # Standard learning rate
        n_steps=1024,            # Reset to 1024
        batch_size=64,           # Reset to 64
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_circuit_breaker_tensorboard/"
    )
    
    print("\n🚀 Starting training with SIMPLIFIED config:")
    print(f"   - Total timesteps: 300,000")
    print(f"   - n_steps: 1024")
    print(f"   - batch_size: 64")
    print(f"   - learning_rate: 3e-4")
    print(f"   - max_steps per episode: 50")
    print(f"   - Observation: 7 features (no history)")
    print(f"\n📊 REWARD SIGNAL:")
    print(f"   - Good state (err<0.1 & p99<500): +10")
    print(f"   - Bad state (err>0.3): -20")
    print(f"   - SLO violation (p99>500): -15")
    print(f"   - False positive (CB open, err<0.05): -5")
    print(f"   - Throughput bonus: rps/20\n")
    
    # Train with 300k timesteps
    model.learn(
        total_timesteps=500000,
        callback=[reward_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save("ppo_circuit_breaker")
    print("\n✅ Model saved: ppo_circuit_breaker.zip")
    print("✅ Best model saved: ./logs/best_model/best_model.zip")
    
    return model, reward_callback


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_reward_curve(callback):
    """Plot training reward curve"""
    
    rewards = callback.episode_rewards
    
    if len(rewards) == 0:
        print("⚠️  No episodes completed!")
        return
    
    # Calculate moving average
    window = min(50, max(1, len(rewards) // 10))
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Raw rewards
    ax1.plot(rewards, alpha=0.3, color='steelblue', label='Episode Reward')
    ax1.plot(range(window-1, len(rewards)), moving_avg, 
             color='darkblue', linewidth=2, label=f'Moving Avg (window={window})')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('PPO Training - Episode Rewards', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Reward distribution
    ax2.hist(rewards, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    ax2.axvline(np.median(rewards), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(rewards):.2f}')
    ax2.set_xlabel('Episode Reward', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Reward Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('PPO Circuit Breaker Training (Simplified Config)', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ppo_reward_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print("TRAINING STATISTICS")
    print("=" * 70)
    print(f"Total Episodes:  {len(rewards)}")
    print(f"Mean Reward:     {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Min Reward:      {np.min(rewards):.2f}")
    print(f"Max Reward:      {np.max(rewards):.2f}")
    print(f"Median Reward:   {np.median(rewards):.2f}")
    print("=" * 70)
    print("✅ Plot saved: ppo_reward_curve.png\n")


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, env, n_episodes=10):
    """Evaluate trained model"""
    
    print("\n" + "=" * 70)
    print(f"EVALUATING MODEL - {n_episodes} episodes")
    print("=" * 70)
    
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep+1:2d}: Reward = {episode_reward:7.2f}, Steps = {steps:3d}")
    
    print("=" * 70)
    print(f"Mean Reward:  {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Length:  {np.mean(episode_lengths):.1f} steps")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Train PPO with simplified config
    model, callback = train_ppo()
    
    # Plot reward curve
    plot_reward_curve(callback)
    
    # Evaluate final model
    env = CircuitBreakerEnv(data_path='data/behavioral_dataset.csv')
    evaluate_model(model, env, n_episodes=10)
    
    # Load and evaluate best model
    print("\n" + "=" * 70)
    print("EVALUATING BEST MODEL (from evaluation callback)")
    print("=" * 70)
    try:
        best_model = PPO.load("./logs/best_model/best_model")
        evaluate_model(best_model, env, n_episodes=10)
    except:
        print("⚠️  Best model not found, skipping best model evaluation")
    
    print("\n🎉 Training complete! Files saved:")
    print("   - ppo_circuit_breaker.zip (final model)")
    print("   - ./logs/best_model/best_model.zip (best model)")
    print("   - ppo_reward_curve.png (plot)")
    print("   - ppo_circuit_breaker_tensorboard/ (logs)")
    print("   - ./logs/ (evaluation results)\n")
