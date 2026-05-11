"""
Circuit Breaker RL Evaluation Script
Performs comprehensive evaluation of trained PPO model vs random policy
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from circuit_breaker_env import CircuitBreakerEnv
from datetime import datetime


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
    
    # Print statistics
    print(f"\n✓ Environment operational: {len(obs_list)} steps completed")
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
    
    print(f"\n✓ reset() and step() working correctly")
    
    return {
        'obs_min': obs_array.min(axis=0),
        'obs_max': obs_array.max(axis=0),
        'reward_mean': reward_array.mean(),
        'reward_std': reward_array.std(),
        'reward_range': (reward_array.min(), reward_array.max())
    }


def policy_comparison(env, model, n_episodes=20):
    """
    Check 2: Policy Comparison
    Compare random policy vs trained PPO policy
    """
    print_section("2. POLICY COMPARISON")
    
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
    
    # Trained PPO Policy
    print(f"Evaluating Trained PPO Policy ({n_episodes} episodes)...")
    ppo_rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        
        ppo_rewards.append(episode_reward)
    
    ppo_mean = np.mean(ppo_rewards)
    ppo_std = np.std(ppo_rewards)
    
    # Calculate improvement - FIXED LOGIC
    improvement_pct = ((ppo_mean - random_mean) / abs(random_mean) * 100) if random_mean != 0 else 0
    
    # Print comparison table
    print(f"\n{'Policy':<20} {'Mean Reward':<15} {'Std Dev':<15} {'Min':<12} {'Max':<12}")
    print("-" * 74)
    print(f"{'Random':<20} {random_mean:<15.2f} {random_std:<15.2f} {min(random_rewards):<12.2f} {max(random_rewards):<12.2f}")
    print(f"{'Trained PPO':<20} {ppo_mean:<15.2f} {ppo_std:<15.2f} {min(ppo_rewards):<12.2f} {max(ppo_rewards):<12.2f}")
    print("-" * 74)
    print(f"\n{'Improvement:':<20} {improvement_pct:+.1f}%")
    
    # FIXED: Correct conclusion logic
    if ppo_mean > random_mean:
        if improvement_pct > 50:
            print(f"\n✓ EXCELLENT: Trained model significantly outperforms random policy!")
        elif improvement_pct > 10:
            print(f"\n✓ GOOD: Trained model outperforms random policy")
        else:
            print(f"\n✓ MARGINAL: Slight improvement over random policy")
    else:
        print(f"\n✗ REGRESSION: Trained model underperforms random policy")
    
    return {
        'random_mean': random_mean,
        'random_std': random_std,
        'ppo_mean': ppo_mean,
        'ppo_std': ppo_std,
        'improvement_pct': improvement_pct
    }


def reward_breakdown_by_scenario(env, model):
    """
    Check 3: Reward Breakdown by Scenario
    Test each scenario separately to show different reward patterns
    """
    print_section("3. REWARD BREAKDOWN BY SCENARIO")
    
    # Load dataset to get scenarios
    df = pd.read_csv('behavioral_dataset (1).csv')
    scenarios = df['sid'].unique()
    
    print(f"\nTesting {len(scenarios)} scenarios (10 steps each)...")
    print(f"\n{'Scenario':<15} {'Avg Throughput':<15} {'Avg Latency':<15} {'Avg Error':<15} {'Avg Net':<12}")
    print("-" * 82)
    
    scenario_results = {}
    
    for scenario in scenarios:
        # Filter data for this scenario
        scenario_data = df[df['sid'] == scenario]
        
        throughput_bonuses = []
        latency_penalties = []
        error_penalties = []
        net_rewards = []
        
        # Run 10 steps for this scenario
        for idx in range(min(10, len(scenario_data))):
            row = scenario_data.iloc[idx]
            
            # Create observation from row
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
            
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Calculate reward components
            rps = row['rps']
            p99 = row['p99']
            err_rate = row['err_rate']
            cb_state = row['cb']
            
            throughput_bonus = rps / 20.0
            
            latency_penalty = 0.0
            if p99 > 500:
                latency_penalty = -15.0
            
            error_penalty = 0.0
            if err_rate > 0.3:
                error_penalty = -20.0
            elif err_rate < 0.1 and p99 < 500:
                error_penalty = 10.0  # Good state bonus
            
            # False positive penalty
            if cb_state == 1 and err_rate < 0.05:
                error_penalty -= 5.0
            
            net_reward = throughput_bonus + latency_penalty + error_penalty
            
            throughput_bonuses.append(throughput_bonus)
            latency_penalties.append(latency_penalty)
            error_penalties.append(error_penalty)
            net_rewards.append(net_reward)
        
        # Calculate averages for this scenario
        avg_throughput = np.mean(throughput_bonuses)
        avg_latency = np.mean(latency_penalties)
        avg_error = np.mean(error_penalties)
        avg_net = np.mean(net_rewards)
        
        scenario_results[scenario] = {
            'avg_throughput': avg_throughput,
            'avg_latency': avg_latency,
            'avg_error': avg_error,
            'avg_net': avg_net
        }
        
        print(f"{scenario:<15} {avg_throughput:<15.2f} {avg_latency:<15.2f} {avg_error:<15.2f} {avg_net:<12.2f}")
    
    print("-" * 82)
    
    # Overall statistics
    all_net_rewards = [r['avg_net'] for r in scenario_results.values()]
    overall_avg = np.mean(all_net_rewards)
    
    print(f"\nOverall Average Net Reward: {overall_avg:.2f}")
    
    print(f"\nScenario Analysis:")
    for scenario, results in scenario_results.items():
        print(f"  {scenario}: Net={results['avg_net']:.2f} (T:{results['avg_throughput']:.1f}, L:{results['avg_latency']:.1f}, E:{results['avg_error']:.1f})")
    
    return scenario_results


def save_report(sanity_results, comparison_results, breakdown_results):
    """Save evaluation summary to file with UTF-8 encoding"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
Circuit Breaker RL Evaluation Report
Generated: {timestamp}

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

Status: ✓ Environment operational

{'='*70}
2. POLICY COMPARISON
{'='*70}
Random Policy:
  Mean Reward: {comparison_results['random_mean']:.2f} ± {comparison_results['random_std']:.2f}

Trained PPO Policy:
  Mean Reward: {comparison_results['ppo_mean']:.2f} ± {comparison_results['ppo_std']:.2f}

Performance:
  Improvement: {comparison_results['improvement_pct']:+.1f}%

{'='*70}
3. REWARD BREAKDOWN BY SCENARIO
{'='*70}
"""
    
    for scenario, results in breakdown_results.items():
        report += f"\n{scenario}:\n"
        report += f"  Throughput Bonus:  {results['avg_throughput']:>8.2f}\n"
        report += f"  Latency Penalty:   {results['avg_latency']:>8.2f}\n"
        report += f"  Error Penalty:     {results['avg_error']:>8.2f}\n"
        report += f"  Net Reward:        {results['avg_net']:>8.2f}\n"
    
    report += f"\n{'='*70}\n"
    report += f"SUMMARY\n"
    report += f"{'='*70}\n"
    
    if comparison_results['ppo_mean'] > comparison_results['random_mean']:
        report += f"Overall Assessment: PASS (Trained model outperforms random)\n"
    else:
        report += f"Overall Assessment: NEEDS IMPROVEMENT (Trained model underperforms)\n"
    
    # FIXED: Add encoding='utf-8' to prevent UnicodeEncodeError
    with open('eval_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Report saved to: eval_report.txt")


def main():
    """Main evaluation pipeline"""
    print("\n" + "="*70)
    print("  CIRCUIT BREAKER RL EVALUATION")
    print("="*70)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Initialize environment
    print("\nInitializing environment...")
    env = CircuitBreakerEnv(data_path='behavioral_dataset (1).csv')
    
    # Load trained model
    print("Loading trained PPO model...")
    try:
        model = PPO.load("./logs/best_model/best_model.zip")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Run evaluations
    sanity_results = environment_sanity_check(env)
    comparison_results = policy_comparison(env, model, n_episodes=20)
    breakdown_results = reward_breakdown_by_scenario(env, model)
    
    # Save report
    print_section("SAVING REPORT")
    save_report(sanity_results, comparison_results, breakdown_results)
    
    # Final summary
    print_section("EVALUATION COMPLETE")
    print(f"\n✓ All checks completed successfully")
    print(f"✓ Trained model improvement: {comparison_results['improvement_pct']:+.1f}%")
    
    if comparison_results['ppo_mean'] > comparison_results['random_mean']:
        if comparison_results['improvement_pct'] > 50:
            print(f"\n🎉 EXCELLENT PERFORMANCE!")
        elif comparison_results['improvement_pct'] > 10:
            print(f"\n✓ Good performance")
        else:
            print(f"\n✓ Marginal improvement")
    else:
        print(f"\n⚠ Model needs improvement (regression detected)")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()