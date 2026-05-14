"""
Fair PPO vs DQN Comparison Script
Compares both algorithms using moving averages of episode rewards
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
import sys
sys.path.append('src')
from circuit_breaker_env import CircuitBreakerEnv
import os


def evaluate_model(model, env, n_episodes=1000):
    """
    Evaluate a trained model and return episode rewards
    
    Args:
        model: Trained RL model (PPO or DQN)
        env: Environment to evaluate on
        n_episodes: Number of episodes to run
        
    Returns:
        List of episode rewards
    """
    episode_rewards = []
    
    print(f"Evaluating model for {n_episodes} episodes...")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        
        if (ep + 1) % 50 == 0:
            print(f"  Completed {ep + 1}/{n_episodes} episodes")
    
    return episode_rewards


def calculate_moving_average(rewards, window=50):
    """
    Calculate moving average of rewards
    
    Args:
        rewards: List of episode rewards
        window: Window size for moving average
        
    Returns:
        Moving average array
    """
    if len(rewards) < window:
        window = max(1, len(rewards))
    
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    return moving_avg


def plot_comparison(ppo_rewards, dqn_rewards, window=50):
    """
    Plot PPO vs DQN comparison with moving averages
    
    Args:
        ppo_rewards: List of PPO episode rewards
        dqn_rewards: List of DQN episode rewards
        window: Window size for moving average
    """
    # Calculate moving averages
    ppo_ma = calculate_moving_average(ppo_rewards, window)
    dqn_ma = calculate_moving_average(dqn_rewards, window)
    
    # Calculate statistics
    ppo_mean = np.mean(ppo_rewards)
    dqn_mean = np.mean(dqn_rewards)
    ppo_std = np.std(ppo_rewards)
    dqn_std = np.std(dqn_rewards)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot PPO moving average
    ppo_x = range(window-1, len(ppo_rewards))
    ax.plot(ppo_x, ppo_ma, color='blue', linewidth=2.5, 
            label=f'PPO Moving Avg (window={window})', alpha=0.9)
    
    # Plot DQN moving average
    dqn_x = range(window-1, len(dqn_rewards))
    ax.plot(dqn_x, dqn_ma, color='red', linewidth=2.5, 
            label=f'DQN Moving Avg (window={window})', alpha=0.9)
    
    # Plot mean lines
    ax.axhline(ppo_mean, color='blue', linestyle='--', linewidth=2, 
               label=f'PPO Mean: {ppo_mean:.2f} ± {ppo_std:.2f}', alpha=0.7)
    ax.axhline(dqn_mean, color='red', linestyle='--', linewidth=2, 
               label=f'DQN Mean: {dqn_mean:.2f} ± {dqn_std:.2f}', alpha=0.7)
    
    # Add shaded regions for standard deviation
    ax.axhspan(ppo_mean - ppo_std, ppo_mean + ppo_std, 
               alpha=0.1, color='blue')
    ax.axhspan(dqn_mean - dqn_std, dqn_mean + dqn_std, 
               alpha=0.1, color='red')
    
    # Labels and title
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Episode Reward', fontsize=13, fontweight='bold')
    ax.set_title('PPO vs DQN - Moving Average Comparison', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add text box with winner
    winner = 'PPO' if ppo_mean > dqn_mean else 'DQN'
    improvement = abs(ppo_mean - dqn_mean)
    textstr = f'Winner: {winner}\nImprovement: {improvement:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/ppo_vs_dqn_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: results/ppo_vs_dqn_comparison.png")
    
    plt.show()


def main():
    """Main comparison function"""
    
    print("=" * 70)
    print("PPO vs DQN - Fair Comparison with Moving Averages")
    print("=" * 70)
    
    # Create environment
    env = CircuitBreakerEnv(data_path='data/behavioral_dataset.csv')
    
    # Load PPO model
    print("\n📘 Loading PPO model...")
    try:
        ppo_model = PPO.load("ppo_circuit_breaker")
        print("✅ PPO model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading PPO model: {e}")
        return
    
    # Load DQN model
    print("\n📕 Loading DQN model...")
    try:
        dqn_model = DQN.load("dqn_circuit_breaker")
        print("✅ DQN model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading DQN model: {e}")
        return
    
    # Evaluate PPO
    print("\n" + "=" * 70)
    print("EVALUATING PPO MODEL")
    print("=" * 70)
    ppo_rewards = evaluate_model(ppo_model, env, n_episodes=1000)
    
    # Evaluate DQN
    print("\n" + "=" * 70)
    print("EVALUATING DQN MODEL")
    print("=" * 70)
    dqn_rewards = evaluate_model(dqn_model, env, n_episodes=1000)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("COMPARISON STATISTICS")
    print("=" * 70)
    print(f"PPO Episodes:     {len(ppo_rewards)}")
    print(f"PPO Mean Reward:  {np.mean(ppo_rewards):.2f} ± {np.std(ppo_rewards):.2f}")
    print(f"PPO Min/Max:      {np.min(ppo_rewards):.2f} / {np.max(ppo_rewards):.2f}")
    print(f"PPO Median:       {np.median(ppo_rewards):.2f}")
    print()
    print(f"DQN Episodes:     {len(dqn_rewards)}")
    print(f"DQN Mean Reward:  {np.mean(dqn_rewards):.2f} ± {np.std(dqn_rewards):.2f}")
    print(f"DQN Min/Max:      {np.min(dqn_rewards):.2f} / {np.max(dqn_rewards):.2f}")
    print(f"DQN Median:       {np.median(dqn_rewards):.2f}")
    print()
    
    # Determine winner
    ppo_mean = np.mean(ppo_rewards)
    dqn_mean = np.mean(dqn_rewards)
    winner = 'PPO' if ppo_mean > dqn_mean else 'DQN'
    improvement = abs(ppo_mean - dqn_mean)
    improvement_pct = (improvement / max(abs(float(ppo_mean)), abs(float(dqn_mean)))) * 100
    
    print(f"🏆 Winner: {winner}")
    print(f"📊 Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")
    print("=" * 70)
    
    # Plot comparison
    print("\n📊 Creating comparison plot...")
    plot_comparison(ppo_rewards, dqn_rewards, window=50)
    
    print("\n🎉 Comparison complete!")
    print("   - Both models evaluated for 200 episodes")
    print("   - Moving averages calculated (window=50)")
    print("   - Comparison plot saved to results/ppo_vs_dqn_comparison.png")


if __name__ == "__main__":
    main()