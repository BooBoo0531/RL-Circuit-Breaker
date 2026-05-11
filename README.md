# RL Circuit Breaker

Reinforcement Learning project for circuit breaker optimization.

## Project Structure

```
RL_Circuit_Breaker/
├── src/                    # Source code
│   ├── circuit_breaker_env.py
│   ├── train.py
│   └── evaluate.py
├── data/                   # Dataset files
│   └── behavioral_dataset.csv
├── models/                 # Trained models
│   └── best_model.zip
├── results/                # Training and evaluation results
│   ├── ppo_reward_curve.png
│   └── eval_report.txt
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python src/train.py
```

### Evaluation
```bash
python src/evaluate.py
```

## Requirements

- Python 3.8+
- See requirements.txt for dependencies
