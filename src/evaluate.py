"""
Circuit Breaker RL Evaluation Script
Performs comprehensive evaluation of trained PPO/DQN models vs random policy
Uses the same reward function as CircuitBreakerEnv for consistency
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN
from circuit_breaker_env import CircuitBreakerEnv
from datetime import datetime


DATA_PATH = 'data/behavioral_dataset.csv'


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def environment_sanity_check(env):
    """
    Check 1: Environment Sanity Check
    Run 100 random steps and verify environment works correctly
    """
    print_section("1. ENVIRONMENT SANITY CHECK")

    obs_list = []
    reward_list = []

    print("Running 100 random steps...")
    obs, _ = env.reset()

    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        obs_list.append(obs)
        reward_list.append(reward)

        if terminated or truncated:
            obs, _ = env.reset()

    obs_array = np.array(obs_list)
    reward_array = np.array(reward_list)

    print(f"\n  Environment operational: {len(obs_list)} steps completed")
    print(f"\nObservation Statistics:")
    print(f"  Shape: {obs_array.shape}")
    print(f"  Min values: {obs_array.min(axis=0)}")
    print(f"  Max values: {obs_array.max(axis=0)}")
    print(f"  Mean values: {obs_array.mean(axis=0)}")

    print(f"\nReward Statistics:")
    print(f"  Mean: {reward_array.mean():.2f}")
    print(f"  Std:  {reward_array.std():.2f}")
    print(f"  Min:  {reward_array.min():.2f}")
    print(f"  Max:  {reward_array.max():.2f}")

    return {
        'obs_min': obs_array.min(axis=0),
        'obs_max': obs_array.max(axis=0),
        'reward_mean': reward_array.mean(),
        'reward_std': reward_array.std(),
        'reward_range': (reward_array.min(), reward_array.max())
    }


def policy_comparison(env, model, model_name="PPO", n_episodes=20):
    """
    Check 2: Policy Comparison
    Compare random policy vs trained policy
    """
    print_section(f"2. POLICY COMPARISON (Random vs {model_name})")

    # Random Policy
    print(f"\nEvaluating Random Policy ({n_episodes} episodes)...")
    random_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        random_rewards.append(episode_reward)

    random_mean = np.mean(random_rewards)
    random_std = np.std(random_rewards)

    # Trained Policy
    print(f"Evaluating Trained {model_name} Policy ({n_episodes} episodes)...")
    trained_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        trained_rewards.append(episode_reward)

    trained_mean = np.mean(trained_rewards)
    trained_std = np.std(trained_rewards)

    improvement_pct = ((trained_mean - random_mean) / abs(random_mean) * 100) if random_mean != 0 else 0

    print(f"\n{'Policy':<20} {'Mean Reward':<15} {'Std Dev':<15} {'Min':<12} {'Max':<12}")
    print("-" * 74)
    print(f"{'Random':<20} {random_mean:<15.2f} {random_std:<15.2f} {min(random_rewards):<12.2f} {max(random_rewards):<12.2f}")
    print(f"{model_name:<20} {trained_mean:<15.2f} {trained_std:<15.2f} {min(trained_rewards):<12.2f} {max(trained_rewards):<12.2f}")
    print("-" * 74)
    print(f"\n{'Improvement:':<20} {improvement_pct:+.1f}%")

    if trained_mean > random_mean:
        if improvement_pct > 50:
            print(f"\n  EXCELLENT: Trained model significantly outperforms random policy!")
        elif improvement_pct > 10:
            print(f"\n  GOOD: Trained model outperforms random policy")
        else:
            print(f"\n  MARGINAL: Slight improvement over random policy")
    else:
        print(f"\n  REGRESSION: Trained model underperforms random policy")

    return {
        'random_mean': random_mean,
        'random_std': random_std,
        'trained_mean': trained_mean,
        'trained_std': trained_std,
        'improvement_pct': improvement_pct
    }


def reward_breakdown_by_scenario(env, model):
    """
    Check 3: Reward Breakdown by Scenario
    Uses CircuitBreakerEnv._calculate_reward for consistent evaluation
    """
    print_section("3. REWARD BREAKDOWN BY SCENARIO")

    df = pd.read_csv(DATA_PATH)
    scenarios = sorted(df['sid'].unique())

    print(f"\nTesting {len(scenarios)} scenarios (10 steps each)...")
    print(f"\n{'Scenario':<12} {'Throughput':<12} {'Latency':<12} {'Error':<12} {'FP':<10} {'Action':<10} {'Net':<10}")
    print("-" * 78)

    scenario_results = {}

    for scenario in scenarios:
        scenario_data = df[df['sid'] == scenario]

        throughput_list = []
        latency_list = []
        error_list = []
        fp_list = []
        action_bonus_list = []
        net_list = []

        for idx in range(min(10, len(scenario_data))):
            row = scenario_data.iloc[idx]

            obs = np.array([
                row['rps'] / 1000.0,
                row['p50'] / 1000.0,
                row['p90'] / 1000.0,
                row['p99'] / 1000.0,
                row['err_rate'],
                row['cb'],
                row['rt_rate']
            ], dtype=np.float32)
            obs = np.clip(obs, 0.0, 1.0)

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            rps = row['rps']
            p99 = row['p99']
            err_rate = row['err_rate']
            cb = row['cb']

            throughput = min(rps / 20.0, 5.0)
            latency = -5.0 if p99 > 500 else 0.0
            error = -10.0 * err_rate
            fp = -3.0 if (cb == 1 and err_rate < 0.05) else 0.0

            if err_rate > 0.3 and action <= 1:
                action_bonus = +2.0
            elif err_rate < 0.05 and action >= 3:
                action_bonus = +2.0
            else:
                action_bonus = 0.0

            net = throughput + latency + error + fp + action_bonus

            throughput_list.append(throughput)
            latency_list.append(latency)
            error_list.append(error)
            fp_list.append(fp)
            action_bonus_list.append(action_bonus)
            net_list.append(net)

        avg_throughput = np.mean(throughput_list)
        avg_latency = np.mean(latency_list)
        avg_error = np.mean(error_list)
        avg_fp = np.mean(fp_list)
        avg_action = np.mean(action_bonus_list)
        avg_net = np.mean(net_list)

        scenario_results[scenario] = {
            'avg_throughput': avg_throughput,
            'avg_latency': avg_latency,
            'avg_error': avg_error,
            'avg_fp': avg_fp,
            'avg_action_bonus': avg_action,
            'avg_net': avg_net
        }

        print(f"  sid={scenario:<5} {avg_throughput:<12.2f} {avg_latency:<12.2f} {avg_error:<12.2f} {avg_fp:<10.2f} {avg_action:<10.2f} {avg_net:<10.2f}")

    print("-" * 78)

    all_net = [r['avg_net'] for r in scenario_results.values()]
    print(f"\nOverall Average Net Reward: {np.mean(all_net):.2f}")

    return scenario_results


def save_report(sanity_results, comparison_results, breakdown_results, model_name="PPO"):
    """Save evaluation summary to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""Circuit Breaker RL Evaluation Report
Generated: {timestamp}
Model: {model_name}

{'='*70}
1. ENVIRONMENT SANITY CHECK
{'='*70}
Observation Range:
  Min: {sanity_results['obs_min']}
  Max: {sanity_results['obs_max']}

Reward Statistics:
  Mean: {sanity_results['reward_mean']:.2f}
  Std:  {sanity_results['reward_std']:.2f}
  Range: [{sanity_results['reward_range'][0]:.2f}, {sanity_results['reward_range'][1]:.2f}]

{'='*70}
2. POLICY COMPARISON
{'='*70}
Random Policy:
  Mean Reward: {comparison_results['random_mean']:.2f} +/- {comparison_results['random_std']:.2f}

Trained {model_name} Policy:
  Mean Reward: {comparison_results['trained_mean']:.2f} +/- {comparison_results['trained_std']:.2f}

Improvement: {comparison_results['improvement_pct']:+.1f}%

{'='*70}
3. REWARD BREAKDOWN BY SCENARIO
{'='*70}
"""

    for scenario, results in breakdown_results.items():
        report += f"\nScenario {scenario}:\n"
        report += f"  Throughput:    {results['avg_throughput']:>8.2f}\n"
        report += f"  Latency:      {results['avg_latency']:>8.2f}\n"
        report += f"  Error:        {results['avg_error']:>8.2f}\n"
        report += f"  False Pos:    {results['avg_fp']:>8.2f}\n"
        report += f"  Action Bonus: {results['avg_action_bonus']:>8.2f}\n"
        report += f"  Net Reward:   {results['avg_net']:>8.2f}\n"

    report += f"\n{'='*70}\n"
    report += f"SUMMARY\n"
    report += f"{'='*70}\n"

    if comparison_results['trained_mean'] > comparison_results['random_mean']:
        report += f"Assessment: PASS (Trained model outperforms random by {comparison_results['improvement_pct']:+.1f}%)\n"
    else:
        report += f"Assessment: NEEDS IMPROVEMENT (Trained model underperforms random)\n"

    os.makedirs('results', exist_ok=True)
    filepath = f'results/eval_report_{model_name.lower()}.txt'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {filepath}")


import os


def main():
    """Main evaluation pipeline"""
    print("\n" + "="*70)
    print("  CIRCUIT BREAKER RL EVALUATION")
    print("="*70)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Initialize environment
    print("\nInitializing environment...")
    env = CircuitBreakerEnv(data_path=DATA_PATH)

    # Run environment sanity check
    sanity_results = environment_sanity_check(env)

    # Evaluate PPO
    print("\n\nLoading PPO model...")
    try:
        ppo_model = PPO.load("ppo_circuit_breaker")
        print("  PPO model loaded (ppo_circuit_breaker.zip)")
        ppo_comparison = policy_comparison(env, ppo_model, model_name="PPO", n_episodes=20)
        ppo_breakdown = reward_breakdown_by_scenario(env, ppo_model)
        save_report(sanity_results, ppo_comparison, ppo_breakdown, model_name="PPO")
    except Exception as e:
        print(f"  PPO model not found: {e}")
        ppo_comparison = None

    # Evaluate DQN
    print("\n\nLoading DQN model...")
    try:
        dqn_model = DQN.load("dqn_circuit_breaker")
        print("  DQN model loaded (dqn_circuit_breaker.zip)")
        dqn_comparison = policy_comparison(env, dqn_model, model_name="DQN", n_episodes=20)
        dqn_breakdown = reward_breakdown_by_scenario(env, dqn_model)
        save_report(sanity_results, dqn_comparison, dqn_breakdown, model_name="DQN")
    except Exception as e:
        print(f"  DQN model not found: {e}")
        dqn_comparison = None

    # Final comparison
    print_section("FINAL COMPARISON")

    if ppo_comparison and dqn_comparison:
        print(f"\n{'Algorithm':<12} {'Mean Reward':<15} {'vs Random':<15}")
        print("-" * 42)
        print(f"{'PPO':<12} {ppo_comparison['trained_mean']:<15.2f} {ppo_comparison['improvement_pct']:+.1f}%")
        print(f"{'DQN':<12} {dqn_comparison['trained_mean']:<15.2f} {dqn_comparison['improvement_pct']:+.1f}%")
        print("-" * 42)
        winner = "PPO" if ppo_comparison['trained_mean'] > dqn_comparison['trained_mean'] else "DQN"
        print(f"\nBest performer: {winner}")
    elif ppo_comparison:
        print(f"\nOnly PPO evaluated. Mean reward: {ppo_comparison['trained_mean']:.2f}")
    elif dqn_comparison:
        print(f"\nOnly DQN evaluated. Mean reward: {dqn_comparison['trained_mean']:.2f}")
    else:
        print("\nNo trained models found. Train PPO or DQN first.")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
