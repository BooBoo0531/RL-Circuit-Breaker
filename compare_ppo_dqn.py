"""
Fair PPO vs DQN Comparison Script
Evaluate PPO and DQN on balanced scenarios and generate comparison plots.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN

sys.path.append('src')
from circuit_breaker_env import CircuitBreakerEnv


DATA_PATH = 'data/behavioral_dataset.csv'
PPO_MODEL_PATH = 'ppo_circuit_breaker'
DQN_MODEL_PATH = 'dqn_circuit_breaker'
RESULTS_DIR = 'results'

N_EPISODES = 1000
MOVING_AVG_WINDOW = 50
SCENARIOS = [1, 2, 3, 4, 5]


def evaluate_model_balanced(model, env, n_episodes=1000):
    """
    Evaluate model with balanced scenario sequence.
    Each scenario appears equally often.
    """
    rewards = []
    scenario_ids = []

    scenario_sequence = [SCENARIOS[i % len(SCENARIOS)] for i in range(n_episodes)]

    for ep, scenario in enumerate(scenario_sequence, start=1):
        obs, info = env.reset(scenario=scenario)

        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        rewards.append(total_reward)
        scenario_ids.append(scenario)

        if ep % 100 == 0:
            print(f'  Completed {ep}/{n_episodes} episodes')

    return np.array(rewards, dtype=float), np.array(scenario_ids, dtype=int)


def moving_average(values, window):
    values = np.array(values, dtype=float)

    if len(values) < window:
        window = len(values)

    return np.convolve(values, np.ones(window) / window, mode='valid')


def print_summary(name, rewards):
    print(f'\n{name} Summary')
    print('-' * 60)
    print(f'Episodes:      {len(rewards)}')
    print(f'Mean reward:   {np.mean(rewards):.2f}')
    print(f'Std reward:    {np.std(rewards):.2f}')
    print(f'Median reward: {np.median(rewards):.2f}')
    print(f'Min reward:    {np.min(rewards):.2f}')
    print(f'Max reward:    {np.max(rewards):.2f}')


def per_scenario_mean(rewards, scenario_ids):
    result = {}

    for sid in SCENARIOS:
        sid_rewards = rewards[scenario_ids == sid]
        result[sid] = {
            'mean': np.mean(sid_rewards),
            'std': np.std(sid_rewards),
            'count': len(sid_rewards)
        }

    return result


def plot_moving_average(ppo_rewards, dqn_rewards):
    ppo_ma = moving_average(ppo_rewards, MOVING_AVG_WINDOW)
    dqn_ma = moving_average(dqn_rewards, MOVING_AVG_WINDOW)

    x = np.arange(MOVING_AVG_WINDOW, len(ppo_rewards) + 1)

    ppo_mean = np.mean(ppo_rewards)
    dqn_mean = np.mean(dqn_rewards)

    plt.figure(figsize=(12, 5))

    plt.plot(
        x,
        ppo_ma,
        linewidth=2.5,
        label=f'PPO Moving Avg (window={MOVING_AVG_WINDOW})'
    )

    plt.plot(
        x,
        dqn_ma,
        linewidth=2.5,
        label=f'DQN Moving Avg (window={MOVING_AVG_WINDOW})'
    )

    plt.axhline(
        ppo_mean,
        linestyle='--',
        linewidth=1.8,
        label=f'PPO Mean: {ppo_mean:.2f}'
    )

    plt.axhline(
        dqn_mean,
        linestyle='--',
        linewidth=1.8,
        label=f'DQN Mean: {dqn_mean:.2f}'
    )

    diff = ppo_mean - dqn_mean
    text = f'PPO Mean: {ppo_mean:.2f}\nDQN Mean: {dqn_mean:.2f}\nDifference: {diff:.2f}'

    plt.text(
        0.02,
        0.97,
        text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.title('PPO vs DQN - Balanced Evaluation Moving Average')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, 'ppo_vs_dqn_moving_average.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved: {output_path}')


def plot_boxplot(ppo_rewards, dqn_rewards):
    plt.figure(figsize=(8, 5))

    plt.boxplot(
        [ppo_rewards, dqn_rewards],
        labels=['PPO', 'DQN'],
        showmeans=True
    )

    plt.title('PPO vs DQN - Reward Distribution')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, 'ppo_vs_dqn_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved: {output_path}')


def plot_per_scenario(ppo_rewards, dqn_rewards, ppo_sids, dqn_sids):
    ppo_stats = per_scenario_mean(ppo_rewards, ppo_sids)
    dqn_stats = per_scenario_mean(dqn_rewards, dqn_sids)

    labels = [f'S{sid}' for sid in SCENARIOS]
    ppo_means = [ppo_stats[sid]['mean'] for sid in SCENARIOS]
    dqn_means = [dqn_stats[sid]['mean'] for sid in SCENARIOS]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(x - width / 2, ppo_means, width, label='PPO')
    plt.bar(x + width / 2, dqn_means, width, label='DQN')

    plt.title('PPO vs DQN - Mean Reward per Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Mean Total Reward')
    plt.xticks(x, labels)
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, 'ppo_vs_dqn_per_scenario.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved: {output_path}')

    print('\nPer-scenario mean reward')
    print('-' * 60)
    print(f'{"Scenario":<10} {"PPO":>12} {"DQN":>12} {"Diff(PPO-DQN)":>18}')
    for sid in SCENARIOS:
        ppo_mean = ppo_stats[sid]['mean']
        dqn_mean = dqn_stats[sid]['mean']
        diff = ppo_mean - dqn_mean
        print(f'S{sid:<9} {ppo_mean:>12.2f} {dqn_mean:>12.2f} {diff:>18.2f}')


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('=' * 70)
    print('PPO vs DQN - Balanced Evaluation Comparison')
    print('=' * 70)

    env = CircuitBreakerEnv(data_path=DATA_PATH)

    print('\nLoading PPO model...')
    ppo_model = PPO.load(PPO_MODEL_PATH)

    print('Loading DQN model...')
    dqn_model = DQN.load(DQN_MODEL_PATH)

    print(f'\nEvaluating PPO for {N_EPISODES} balanced episodes...')
    ppo_rewards, ppo_sids = evaluate_model_balanced(
        ppo_model,
        env,
        n_episodes=N_EPISODES
    )

    print(f'\nEvaluating DQN for {N_EPISODES} balanced episodes...')
    dqn_rewards, dqn_sids = evaluate_model_balanced(
        dqn_model,
        env,
        n_episodes=N_EPISODES
    )

    print_summary('PPO', ppo_rewards)
    print_summary('DQN', dqn_rewards)

    plot_moving_average(ppo_rewards, dqn_rewards)
    plot_boxplot(ppo_rewards, dqn_rewards)
    plot_per_scenario(ppo_rewards, dqn_rewards, ppo_sids, dqn_sids)

    print('\nComparison complete.')
    print(f'Generated plots in: {RESULTS_DIR}/')
    print(' - ppo_vs_dqn_moving_average.png')
    print(' - ppo_vs_dqn_boxplot.png')
    print(' - ppo_vs_dqn_per_scenario.png')


if __name__ == '__main__':
    main()