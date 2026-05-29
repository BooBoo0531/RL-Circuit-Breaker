"""
Comprehensive Evaluation for RL Circuit Breaker
Evaluates trained PPO/DQN models across 11 metrics in 2 categories:
  Part A: Model Quality (convergence, baselines, action distribution, appropriateness, stats)
  Part B: System Quality (SLO, error rate, throughput, false positives, reaction time)

Usage:
    cd RL-Circuit-Breaker/
    python src/evaluate_comprehensive.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from stable_baselines3 import PPO, DQN
from circuit_breaker_env import CircuitBreakerEnv
from datetime import datetime
import os


DATA_PATH = 'data/behavioral_dataset.csv'
N_EVAL_EPISODES = 50


def print_header(title):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def print_subheader(title):
    print(f"\n--- {title} ---")


# ==========================================================================
# POLICIES
# ==========================================================================

def rule_based_policy(obs):
    """
    Simple heuristic baseline:
    - err_rate > 0.3 → Emergency (0)
    - err_rate > 0.1 → Strict (1)
    - err_rate > 0.05 → Moderate (2)
    - err_rate < 0.05 and p99 < 0.5 (normalized) → Liberal (4)
    - else → Relaxed (3)
    """
    err_rate = obs[4]
    p99_norm = obs[3]

    if err_rate > 0.3:
        return 0
    elif err_rate > 0.1:
        return 1
    elif err_rate > 0.05:
        return 2
    elif p99_norm < 0.5:
        return 4
    else:
        return 3


def random_policy(env):
    return env.action_space.sample()


# ==========================================================================
# PART A: MODEL EVALUATION
# ==========================================================================

def eval_convergence(env, model, model_name, n_windows=5):
    """
    Metric 1: Convergence Analysis
    Run episodes and compare early vs late performance.
    NOTE: This evaluates a fixed trained model deterministically. Any variance
    between windows reflects stochastic scenario selection by the environment,
    NOT ongoing learning or training instability.
    """
    print_header(f"1. CONVERGENCE ANALYSIS ({model_name})")

    n_episodes = N_EVAL_EPISODES
    rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)

    window_size = n_episodes // n_windows
    print(f"\n  Performance across {n_windows} windows ({window_size} episodes each):")
    print(f"  {'Window':<12} {'Mean Reward':<15} {'Std':<12}")
    print(f"  {'-'*39}")

    window_means = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        w_mean = np.mean(rewards[start:end])
        w_std = np.std(rewards[start:end])
        window_means.append(w_mean)
        print(f"  {f'W{i+1} ({start}-{end})':<12} {w_mean:<15.2f} {w_std:<12.2f}")

    first_half = np.mean(rewards[:n_episodes//2])
    second_half = np.mean(rewards[n_episodes//2:])
    print(f"\n  First half mean:  {first_half:.2f}")
    print(f"  Second half mean: {second_half:.2f}")
    print(f"  Note: Since evaluation uses a trained model deterministically,")
    print(f"        variance comes from random scenario selection, not learning.")

    return rewards


def eval_multi_baseline(env, models_dict):
    """
    Metric 2: Multi-baseline Comparison
    Compare Random vs Rule-based vs PPO vs DQN
    """
    print_header("2. MULTI-BASELINE COMPARISON")

    results = {}

    for name, policy in models_dict.items():
        rewards = []
        for _ in range(N_EVAL_EPISODES):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                if name == "Random":
                    action = random_policy(env)
                elif name == "Rule-based":
                    action = rule_based_policy(obs)
                else:
                    action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            rewards.append(episode_reward)

        results[name] = {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'median': np.median(rewards),
            'rewards': rewards
        }

    print(f"\n  {'Policy':<14} {'Mean':<10} {'Std':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
    print(f"  {'-'*64}")
    for name, r in results.items():
        print(f"  {name:<14} {r['mean']:<10.2f} {r['std']:<10.2f} {r['median']:<10.2f} {r['min']:<10.2f} {r['max']:<10.2f}")

    print(f"\n  Improvement vs Random:")
    random_mean = results['Random']['mean']
    for name, r in results.items():
        if name != "Random":
            imp = ((r['mean'] - random_mean) / abs(random_mean) * 100) if random_mean != 0 else 0
            print(f"    {name:<14} {imp:+.1f}%")

    print(f"\n  Improvement vs Rule-based:")
    rule_mean = results['Rule-based']['mean']
    for name, r in results.items():
        if name not in ["Random", "Rule-based"]:
            imp = ((r['mean'] - rule_mean) / abs(rule_mean) * 100) if rule_mean != 0 else 0
            print(f"    {name:<14} {imp:+.1f}%")

    return results


def eval_action_distribution(env, models_dict):
    """
    Metric 3: Action Distribution per Scenario
    Shows what actions each model chooses for each scenario
    """
    print_header("3. ACTION DISTRIBUTION PER SCENARIO")

    df = pd.read_csv(DATA_PATH)
    scenarios = sorted(df['sid'].unique())
    action_names = ['Emergency', 'Strict', 'Moderate', 'Relaxed', 'Liberal']

    for name, policy in models_dict.items():
        if name == "Random":
            continue

        print_subheader(f"{name} - Action Distribution (%)")
        print(f"  {'Scenario':<10}", end="")
        for a_name in action_names:
            print(f"{a_name:<12}", end="")
        print(f"{'Dominant':<12}")
        print(f"  {'-'*70}")

        for sid in scenarios:
            action_counts = np.zeros(5)
            n_steps = 0

            for _ in range(10):
                obs, _ = env.reset()
                # Force specific scenario
                env.current_scenario = sid
                env.scenario_data = env.full_df[env.full_df['sid'] == sid].sample(frac=1.0).reset_index(drop=True)
                obs = env._get_observation(env.scenario_data.iloc[0])

                for step in range(100):
                    if name == "Rule-based":
                        action = rule_based_policy(obs)
                    else:
                        action, _ = policy.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    action_counts[action] += 1
                    n_steps += 1
                    if terminated or truncated:
                        break

            pcts = action_counts / n_steps * 100
            dominant = action_names[np.argmax(pcts)]
            print(f"  sid={sid:<5}", end="")
            for p in pcts:
                print(f"{p:<12.1f}", end="")
            print(f"{dominant:<12}")

        print()


def eval_action_appropriateness(env, models_dict):
    """
    Metric 4: Action Appropriateness Rate
    Measures % of actions that match domain logic:
    - High error (>0.3): should choose Emergency/Strict (0,1)
    - Low error (<0.05): should choose Relaxed/Liberal (3,4)
    - Medium error (0.05-0.3): should choose Strict/Moderate/Relaxed (1,2,3)
    """
    print_header("4. ACTION APPROPRIATENESS RATE")

    results = {}

    for name, policy in models_dict.items():
        if name == "Random":
            continue

        total_high_err = 0
        correct_high_err = 0
        total_low_err = 0
        correct_low_err = 0
        total_mid_err = 0
        correct_mid_err = 0

        for _ in range(N_EVAL_EPISODES):
            obs, _ = env.reset()
            done = False
            while not done:
                if name == "Rule-based":
                    action = rule_based_policy(obs)
                else:
                    action, _ = policy.predict(obs, deterministic=True)

                err_rate = obs[4]

                if err_rate > 0.3:
                    total_high_err += 1
                    if action <= 1:
                        correct_high_err += 1
                elif err_rate < 0.05:
                    total_low_err += 1
                    if action >= 3:
                        correct_low_err += 1
                else:
                    total_mid_err += 1
                    if 1 <= action <= 3:
                        correct_mid_err += 1

                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

        high_rate = (correct_high_err / total_high_err * 100) if total_high_err > 0 else 0
        low_rate = (correct_low_err / total_low_err * 100) if total_low_err > 0 else 0
        mid_rate = (correct_mid_err / total_mid_err * 100) if total_mid_err > 0 else 0
        total = total_high_err + total_low_err + total_mid_err
        correct = correct_high_err + correct_low_err + correct_mid_err
        overall_rate = (correct / total * 100) if total > 0 else 0

        results[name] = {
            'high_err_rate': high_rate,
            'low_err_rate': low_rate,
            'mid_err_rate': mid_rate,
            'overall': overall_rate,
            'total_high': total_high_err,
            'total_low': total_low_err,
            'total_mid': total_mid_err
        }

    print(f"\n  {'Policy':<14} {'High Err':<14} {'Low Err':<14} {'Mid Err':<14} {'Overall':<10}")
    print(f"  {'':14} {'(>0.3→0,1)':<14} {'(<0.05→3,4)':<14} {'(0.05-0.3→1-3)':<14}")
    print(f"  {'-'*66}")
    for name, r in results.items():
        print(f"  {name:<14} {r['high_err_rate']:<14.1f} {r['low_err_rate']:<14.1f} {r['mid_err_rate']:<14.1f} {r['overall']:<10.1f}")

    print(f"\n  Sample sizes (steps):")
    for name, r in results.items():
        print(f"    {name}: high_err={r['total_high']}, low_err={r['total_low']}, mid_err={r['total_mid']}")

    return results


def eval_statistical_significance(baseline_results):
    """
    Metric 5: Statistical Significance
    Welch's t-test and 95% confidence intervals
    """
    print_header("5. STATISTICAL SIGNIFICANCE")

    policies = list(baseline_results.keys())
    rewards_dict = {name: r['rewards'] for name, r in baseline_results.items()}

    # 95% Confidence Intervals
    print_subheader("95% Confidence Intervals")
    print(f"  {'Policy':<14} {'Mean':<10} {'95% CI':<24} {'CI Width':<10}")
    print(f"  {'-'*58}")
    for name, rewards in rewards_dict.items():
        mean = np.mean(rewards)
        ci = stats.t.interval(0.95, len(rewards)-1, loc=mean, scale=stats.sem(rewards))
        width = ci[1] - ci[0]
        print(f"  {name:<14} {mean:<10.2f} [{ci[0]:.2f}, {ci[1]:.2f}]{'':<4} {width:<10.2f}")

    # Pairwise t-tests
    print_subheader("Pairwise Welch's t-test (p-values)")
    comparisons = [
        ("PPO", "Random"),
        ("DQN", "Random"),
        ("PPO", "Rule-based"),
        ("DQN", "Rule-based"),
        ("PPO", "DQN"),
    ]

    print(f"  {'Comparison':<24} {'t-stat':<10} {'p-value':<12} {'Significant?':<14}")
    print(f"  {'-'*60}")
    for a, b in comparisons:
        if a in rewards_dict and b in rewards_dict:
            t_stat, p_val = stats.ttest_ind(rewards_dict[a], rewards_dict[b], equal_var=False)
            sig = "YES (p<0.05)" if p_val < 0.05 else "NO"
            print(f"  {f'{a} vs {b}':<24} {t_stat:<10.3f} {p_val:<12.6f} {sig:<14}")


# ==========================================================================
# PART B: SYSTEM EVALUATION
# ==========================================================================

def eval_system_metrics(env, models_dict):
    """
    Metrics 6-10: System-level evaluation
    Runs episodes and tracks per-step metrics for each policy
    """
    print_header("6-10. SYSTEM QUALITY METRICS")

    system_results = {}

    for name, policy in models_dict.items():
        slo_violations = 0
        total_steps = 0
        err_rates_sum = 0
        rps_sum = 0
        false_positives = 0
        strict_when_healthy = 0
        total_healthy_steps = 0
        reaction_times = []

        for _ in range(N_EVAL_EPISODES):
            obs, _ = env.reset()
            done = False
            prev_action = None
            steps_since_err_spike = None
            reacted = False
            spike_active = False  # tracks whether we are currently inside an error spike

            while not done:
                if name == "Random":
                    action = random_policy(env)
                elif name == "Rule-based":
                    action = rule_based_policy(obs)
                else:
                    action, _ = policy.predict(obs, deterministic=True)

                err_rate = obs[4]
                p99_norm = obs[3]
                rps_norm = obs[0]

                total_steps += 1

                # Metric 6: SLO Compliance (p99 < 500ms → p99_norm < 0.5)
                if p99_norm >= 0.5:
                    slo_violations += 1

                # Metric 7: Error rate tracking
                err_rates_sum += err_rate

                # Metric 8: Throughput
                rps_sum += rps_norm

                # Metric 9: False Positive (strict action when error is low)
                if err_rate < 0.05:
                    total_healthy_steps += 1
                    if action <= 1:
                        false_positives += 1

                # Metric 10: Reaction time
                # A new spike starts when error rises above 0.3 and we are not already
                # tracking one. Once the agent reacts (action <= 1) we record the time
                # and close the spike. The spike also closes when error drops back below 0.3.
                if err_rate > 0.3 and not spike_active:
                    # New spike detected — begin counting steps to reaction
                    spike_active = True
                    steps_since_err_spike = 0
                    reacted = False
                elif err_rate <= 0.3:
                    # Spike has subsided — reset tracking state
                    spike_active = False
                    steps_since_err_spike = None
                    reacted = False

                if spike_active and not reacted:
                    steps_since_err_spike += 1
                    if action <= 1:
                        reaction_times.append(steps_since_err_spike)
                        reacted = True
                        # Keep spike_active=True until error subsides so we
                        # don't double-count a reaction within the same spike.

                prev_action = action
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

        slo_compliance = (1 - slo_violations / total_steps) * 100 if total_steps > 0 else 0
        avg_err_rate = err_rates_sum / total_steps if total_steps > 0 else 0
        avg_throughput = rps_sum / total_steps if total_steps > 0 else 0
        fp_rate = (false_positives / total_healthy_steps * 100) if total_healthy_steps > 0 else 0
        avg_reaction = np.mean(reaction_times) if reaction_times else float('nan')

        system_results[name] = {
            'slo_compliance': slo_compliance,
            'avg_err_rate': avg_err_rate,
            'avg_throughput': avg_throughput,
            'fp_rate': fp_rate,
            'avg_reaction_time': avg_reaction,
            'total_steps': total_steps,
            'n_reactions': len(reaction_times)
        }

    # Print Metric 6: SLO Compliance
    print_subheader("Metric 6: SLO Compliance Rate (p99 < 500ms)")
    print(f"  {'Policy':<14} {'Compliance %':<15}")
    print(f"  {'-'*29}")
    for name, r in system_results.items():
        print(f"  {name:<14} {r['slo_compliance']:<15.1f}")

    # Print Metric 7: Error Rate
    print_subheader("Metric 7: Average Error Rate During Operation")
    print(f"  {'Policy':<14} {'Avg Err Rate':<15} {'vs Random':<12}")
    print(f"  {'-'*41}")
    random_err = system_results['Random']['avg_err_rate']
    for name, r in system_results.items():
        reduction = ((random_err - r['avg_err_rate']) / random_err * 100) if random_err > 0 else 0
        print(f"  {name:<14} {r['avg_err_rate']:<15.4f} {reduction:+.1f}%")

    # Print Metric 8: Throughput
    print_subheader("Metric 8: Throughput Preservation (normalized RPS)")
    print(f"  {'Policy':<14} {'Avg RPS (norm)':<15} {'% of Max':<12}")
    print(f"  {'-'*41}")
    for name, r in system_results.items():
        pct_max = r['avg_throughput'] * 100
        print(f"  {name:<14} {r['avg_throughput']:<15.4f} {pct_max:.1f}%")

    # Print Metric 9: False Positive Rate
    print_subheader("Metric 9: False Positive Rate (strict action when err < 5%)")
    print(f"  {'Policy':<14} {'FP Rate %':<12} {'Interpretation':<30}")
    print(f"  {'-'*56}")
    for name, r in system_results.items():
        interp = "GOOD (< 10%)" if r['fp_rate'] < 10 else ("OK (10-30%)" if r['fp_rate'] < 30 else "HIGH (> 30%)")
        print(f"  {name:<14} {r['fp_rate']:<12.1f} {interp:<30}")

    # Print Metric 10: Reaction Time
    print_subheader("Metric 10: Reaction Time (steps to switch to strict on error spike)")
    print(f"  {'Policy':<14} {'Avg Steps':<12} {'N Reactions':<14} {'Interpretation':<20}")
    print(f"  {'-'*60}")
    for name, r in system_results.items():
        if np.isnan(r['avg_reaction_time']):
            interp = "N/A"
            rt_str = "N/A"
        elif r['avg_reaction_time'] <= 2:
            interp = "FAST"
            rt_str = f"{r['avg_reaction_time']:.1f}"
        elif r['avg_reaction_time'] <= 5:
            interp = "OK"
            rt_str = f"{r['avg_reaction_time']:.1f}"
        else:
            interp = "SLOW"
            rt_str = f"{r['avg_reaction_time']:.1f}"
        print(f"  {name:<14} {rt_str:<12} {r['n_reactions']:<14} {interp:<20}")

    return system_results


def eval_per_scenario_summary(env, models_dict):
    """
    Metric 11: Per-Scenario Performance Summary
    """
    print_header("11. PER-SCENARIO PERFORMANCE SUMMARY")

    df = pd.read_csv(DATA_PATH)
    scenarios = sorted(df['sid'].unique())

    for name, policy in models_dict.items():
        if name == "Random":
            continue

        print_subheader(f"{name} - Per Scenario")
        print(f"  {'Sid':<6} {'Reward':<10} {'SLO%':<8} {'ErrRate':<10} {'FP%':<8} {'Dominant Action':<16}")
        print(f"  {'-'*58}")

        for sid in scenarios:
            rewards = []
            slo_ok = 0
            total = 0
            err_sum = 0
            fp = 0
            healthy = 0
            action_counts = np.zeros(5)

            for _ in range(10):
                obs, _ = env.reset()
                env.current_scenario = sid
                env.scenario_data = env.full_df[env.full_df['sid'] == sid].sample(frac=1.0).reset_index(drop=True)
                obs = env._get_observation(env.scenario_data.iloc[0])
                ep_reward = 0

                for step in range(100):
                    if name == "Rule-based":
                        action = rule_based_policy(obs)
                    else:
                        action, _ = policy.predict(obs, deterministic=True)

                    action_counts[action] += 1
                    err_rate = obs[4]
                    p99_norm = obs[3]
                    total += 1

                    if p99_norm < 0.5:
                        slo_ok += 1
                    err_sum += err_rate
                    if err_rate < 0.05:
                        healthy += 1
                        if action <= 1:
                            fp += 1

                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += reward
                    if terminated or truncated:
                        break

                rewards.append(ep_reward)

            action_names = ['Emergency', 'Strict', 'Moderate', 'Relaxed', 'Liberal']
            dominant = action_names[np.argmax(action_counts)]
            slo_pct = slo_ok / total * 100 if total > 0 else 0
            avg_err = err_sum / total if total > 0 else 0
            fp_pct = fp / healthy * 100 if healthy > 0 else 0

            print(f"  {sid:<6} {np.mean(rewards):<10.2f} {slo_pct:<8.1f} {avg_err:<10.4f} {fp_pct:<8.1f} {dominant:<16}")

        print()


# ==========================================================================
# REPORT GENERATION
# ==========================================================================

def save_comprehensive_report(baseline_results, appropriateness_results, system_results):
    """Save comprehensive report to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = []
    report.append("=" * 74)
    report.append("COMPREHENSIVE EVALUATION REPORT")
    report.append(f"RL for Adaptive Circuit Breaking and Retry Strategy")
    report.append(f"Generated: {timestamp}")
    report.append(f"Episodes per policy: {N_EVAL_EPISODES}")
    report.append("=" * 74)

    # Baseline comparison summary
    report.append("\n\nMULTI-BASELINE COMPARISON")
    report.append("-" * 40)
    for name, r in baseline_results.items():
        report.append(f"  {name:<14} Mean: {r['mean']:.2f} +/- {r['std']:.2f}")

    # Action appropriateness
    report.append("\n\nACTION APPROPRIATENESS (%)")
    report.append("-" * 40)
    for name, r in appropriateness_results.items():
        report.append(f"  {name:<14} Overall: {r['overall']:.1f}%  High-err: {r['high_err_rate']:.1f}%  Low-err: {r['low_err_rate']:.1f}%")

    # System metrics
    report.append("\n\nSYSTEM METRICS")
    report.append("-" * 40)
    for name, r in system_results.items():
        report.append(f"  {name}:")
        report.append(f"    SLO Compliance:  {r['slo_compliance']:.1f}%")
        report.append(f"    Avg Error Rate:  {r['avg_err_rate']:.4f}")
        report.append(f"    Throughput:      {r['avg_throughput']:.4f}")
        report.append(f"    FP Rate:         {r['fp_rate']:.1f}%")
        rt = f"{r['avg_reaction_time']:.1f}" if not np.isnan(r['avg_reaction_time']) else "N/A"
        report.append(f"    Reaction Time:   {rt} steps")

    # Conclusion
    report.append("\n\n" + "=" * 74)
    report.append("CONCLUSION")
    report.append("=" * 74)

    best_policy = max(
        [(name, r['mean']) for name, r in baseline_results.items()],
        key=lambda x: x[1]
    )
    report.append(f"  Best performer: {best_policy[0]} (mean reward: {best_policy[1]:.2f})")

    ppo_better = baseline_results.get('PPO', {}).get('mean', 0) > baseline_results.get('Rule-based', {}).get('mean', 0)
    dqn_better = baseline_results.get('DQN', {}).get('mean', 0) > baseline_results.get('Rule-based', {}).get('mean', 0)

    if ppo_better or dqn_better:
        report.append("  RL models outperform rule-based heuristic → RL approach justified")
    else:
        report.append("  RL models do NOT outperform rule-based → needs improvement")

    report.append("=" * 74)

    # Known limitations
    report.append("\n\n" + "=" * 74)
    report.append("KNOWN LIMITATIONS")
    report.append("=" * 74)
    report.append("  1. TRAIN/EVAL DATASET OVERLAP (EVAL-001):")
    report.append("     Training and evaluation use the same dataset. Metrics reflect")
    report.append("     in-sample performance. Hold-out split (e.g. sid=5 as test) is")
    report.append("     recommended before drawing generalization conclusions.")
    report.append("")
    report.append("  2. HIGH STANDARD DEVIATION FROM SCENARIO SAMPLING:")
    report.append("     Episode reward variance is dominated by which scenario the")
    report.append("     environment randomly selects, not by policy quality. Std values")
    report.append("     should be interpreted with this in mind; per-scenario breakdowns")
    report.append("     (Metric 11) provide more stable signal.")
    report.append("")
    report.append("  3. OFFLINE SIMULATION — NOT PRODUCTION KUBERNETES:")
    report.append("     All results are from a pre-recorded behavioral dataset. Real")
    report.append("     Kubernetes deployments will have one-step feedback delay, noisy")
    report.append("     Prometheus scrape intervals, and actuation latency not captured")
    report.append("     here. Results should be validated against a live staging cluster")
    report.append("     before production use.")
    report.append("=" * 74)

    os.makedirs('results', exist_ok=True)
    filepath = 'results/comprehensive_eval_report.txt'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\n  Report saved: {filepath}")


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    # -----------------------------------------------------------------------
    # MODEL LOADING RATIONALE
    # -----------------------------------------------------------------------
    # PPO  — final model ("ppo_circuit_breaker"):
    #   compare_checkpoints.py found no statistically significant difference
    #   between the PPO best checkpoint and the final model, and the final
    #   model had a marginally higher mean reward, so the final model is used.
    #   Path: ppo_circuit_breaker.zip  (saved by train_ppo.py at end of run)
    #
    # DQN  — best checkpoint ("logs/dqn_best_model/best_model"):
    #   compare_checkpoints.py detected policy degradation in DQN: the best
    #   checkpoint (saved by EvalCallback during training) outperformed the
    #   final model, so the best checkpoint is loaded here.
    #   Path: logs/dqn_best_model/best_model.zip  (saved by EvalCallback)
    #   Fallback: dqn_circuit_breaker.zip (final model) if checkpoint absent.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("  COMPREHENSIVE EVALUATION")
    print("  RL for Adaptive Circuit Breaking and Retry Strategy")
    print("  in Kubernetes Microservices")
    print("=" * 74)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Episodes per policy: {N_EVAL_EPISODES}")
    print("=" * 74)

    # Initialize environment
    env = CircuitBreakerEnv(data_path=DATA_PATH)

    # Load models — PPO: final model wins, DQN: best checkpoint wins (per compare_checkpoints.py)
    models_dict = {'Random': None, 'Rule-based': None}

    print("\n  Loading models...")

    # PPO: use final model (no degradation, final > best checkpoint)
    ppo_final_path = "ppo_circuit_breaker"
    try:
        ppo_model = PPO.load(ppo_final_path)
        models_dict['PPO'] = ppo_model
        print(f"    PPO: loaded final model ({ppo_final_path}) — final wins over best checkpoint")
    except Exception as e:
        print(f"    PPO: not found ({e})")

    # DQN: use best checkpoint (policy degradation detected)
    dqn_best_path = "logs/dqn_best_model/best_model"
    dqn_final_path = "dqn_circuit_breaker"
    try:
        dqn_model = DQN.load(dqn_best_path)
        models_dict['DQN'] = dqn_model
        print(f"    DQN: loaded best checkpoint ({dqn_best_path}) — policy degradation detected")
    except Exception:
        try:
            dqn_model = DQN.load(dqn_final_path)
            models_dict['DQN'] = dqn_model
            print(f"    DQN: best checkpoint not found, loaded final model ({dqn_final_path})")
        except Exception as e:
            print(f"    DQN: not found ({e})")

    # ======================================================================
    # PART A: MODEL EVALUATION
    # ======================================================================
    print("\n\n" + "#" * 74)
    print("  PART A: MODEL QUALITY EVALUATION")
    print("#" * 74)

    # Metric 1: Convergence
    if 'PPO' in models_dict:
        eval_convergence(env, models_dict['PPO'], "PPO")
    if 'DQN' in models_dict:
        eval_convergence(env, models_dict['DQN'], "DQN")

    # Metric 2: Multi-baseline
    baseline_results = eval_multi_baseline(env, models_dict)

    # Metric 3: Action Distribution
    eval_action_distribution(env, models_dict)

    # Metric 4: Action Appropriateness
    appropriateness_results = eval_action_appropriateness(env, models_dict)

    # Metric 5: Statistical Significance
    eval_statistical_significance(baseline_results)

    # ======================================================================
    # PART B: SYSTEM EVALUATION
    # ======================================================================
    print("\n\n" + "#" * 74)
    print("  PART B: SYSTEM QUALITY EVALUATION")
    print("#" * 74)

    # Metrics 6-10: System metrics
    system_results = eval_system_metrics(env, models_dict)

    # Metric 11: Per-scenario summary
    eval_per_scenario_summary(env, models_dict)

    # ======================================================================
    # SAVE REPORT
    # ======================================================================
    save_comprehensive_report(baseline_results, appropriateness_results, system_results)

    print("\n" + "=" * 74)
    print("  EVALUATION COMPLETE")
    print("=" * 74)
    print()


if __name__ == "__main__":
    main()
