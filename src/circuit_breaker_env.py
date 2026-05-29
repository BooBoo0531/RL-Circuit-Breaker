import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CircuitBreakerEnv(gym.Env):
    """Gymnasium environment for Circuit Breaking in Kubernetes.

    Normalization note: max_rps=1000 and max_latency=1000 are calibrated for
    an Azure Standard_B2s_v2 node running the Fortio load generator at ~20 RPS
    steady-state. Real observed values in this experiment stay well below these
    ceilings (max rps ~84, max latency ~900 ms), so normalized features occupy
    only a fraction of the [0, 1] range. Adjusting these constants to match the
    actual dataset range (e.g. max_rps=100) would improve gradient signal.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, data_path='data/behavioral_dataset.csv'):
        super().__init__()
        
        # Load behavioral data
        import pandas as pd
        if isinstance(data_path, pd.DataFrame):
            self.full_df = data_path.sort_values('ts').reset_index(drop=True)
        else:
            self.full_df = pd.read_csv(data_path).sort_values('ts').reset_index(drop=True)
        
        # Action space: 5 actions (0-4)
        # 0=Emergency, 1=Strict, 2=Moderate, 3=Relaxed, 4=Liberal
        self.action_space = spaces.Discrete(5)
        
        # Observation space: 7 features normalized to [0,1]
        # [rps, p50, p90, p99, err_rate, cb_state, retry_rate]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Normalization constants for features
        self.max_rps = 1000.0
        self.max_latency = 1000.0  # for p50, p90, p99
        
        # Episode management
        self.current_step = 0
        self.max_steps = 100
        self.current_scenario = None
        self.scenario_data = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomly pick a scenario (sid 1-5) using gymnasium's seeded RNG
        self.current_scenario = int(self.np_random.integers(1, 6))

        # Load rows for that scenario only
        self.scenario_data = self.full_df[self.full_df['sid'] == self.current_scenario].copy()

        # Shuffle the scenario data using a seed derived from the gymnasium RNG
        shuffle_seed = int(self.np_random.integers(0, 2**31))
        self.scenario_data = self.scenario_data.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
        
        # Start new episode
        self.current_step = 0
        
        # Get initial observation from first row
        if len(self.scenario_data) > 0:
            obs = self._get_observation(self.scenario_data.iloc[0])
        else:
            # Fallback if scenario has no data
            obs = np.zeros(7, dtype=np.float32)
        
        info = {'scenario': self.current_scenario}
        return obs, info
    
    def step(self, action):
        # Cycle through rows of current scenario
        idx = self.current_step % len(self.scenario_data)
        row = self.scenario_data.iloc[idx]
        
        # Get observation with cb_state and retry_rate from dataset
        obs = self._get_observation(row)
        
        # Calculate reward with clear signal
        reward = self._calculate_reward(row, action)
        
        # Increment step
        self.current_step += 1
        
        # Episode termination after max_steps
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            'scenario': self.current_scenario,
            'rps': row['rps'],
            'p99': row['p99'],
            'err_rate': row['err_rate'],
            'cb_state': row['cb'],
            'retry_rate': row['rt_rate']
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self, row):
        """7-feature observation normalized to [0,1]"""
        state = np.array([
            row['rps'] / self.max_rps,           # Normalize rps
            row['p50'] / self.max_latency,       # Normalize p50
            row['p90'] / self.max_latency,       # Normalize p90
            row['p99'] / self.max_latency,       # Normalize p99
            row['err_rate'],                     # Already [0,1]
            row['cb'],                           # cb_state from dataset
            row['rt_rate']                       # retry_rate from dataset
        ], dtype=np.float32)
        # Clip to ensure [0,1] range
        return np.clip(state, 0.0, 1.0)
    
    def _calculate_reward(self, row, action):
        """
        NEW BALANCED REWARD DESIGN:
        All components within 3x of each other to avoid bimodal rewards.
        
        Components:
        - Throughput bonus: 0 to +5 (scaled from rps)
        - Latency penalty: 0 or -5 (binary based on p99 > 500ms)
        - Error penalty: 0 to -10 (scaled linearly with err_rate)
        - False positive penalty: 0 or -3 (CB open when err_rate < 0.05)
        - Action appropriateness bonus: 0 or +2 (reward good action choices)
        
        Range: -18 to +7 per step
        """
        rps = row['rps']
        p99 = row['p99']
        err_rate = row['err_rate']
        cb = row['cb']
        
        # Throughput bonus: 0 to +5
        throughput = min(rps / 20.0, 5.0)
        
        # Latency penalty: 0 or -5
        latency = -5.0 if p99 > 500 else 0.0
        
        # Error penalty: scaled, not binary
        # err_rate 0->1 maps to 0->-10
        error = -10.0 * err_rate
        
        # False positive: -3 if unnecessary CB open
        # NOTE (known limitation, CB-002): `cb` here is the circuit-breaker state
        # recorded in the dataset (environment observation), NOT the action chosen
        # by the agent. Because the training dataset has cb==0 in every row, this
        # penalty never fires and provides no training signal. A future fix should
        # condition on the agent's action (e.g. action <= 1) instead of the dataset
        # cb column to correctly penalise false-positive circuit-breaker trips.
        fp = -3.0 if (cb == 1 and err_rate < 0.05) else 0.0
        
        # Action appropriateness bonus
        # Action 0,1 (strict) good when err high
        # Action 3,4 (liberal) good when err low
        if err_rate > 0.3 and action <= 1:
            action_bonus = +2.0
        elif err_rate < 0.05 and action >= 3:
            action_bonus = +2.0
        else:
            action_bonus = 0.0
        
        return throughput + latency + error + fp + action_bonus
    
    def render(self):
        pass
    
    def close(self):
        pass