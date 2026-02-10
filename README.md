# Transformer Reasoning Research

**Research Project**: Investigating emergent reasoning capabilities in transformer models through chain-of-thought prompting and autonomous scratch space utilization.

**Researcher**: Mubashshir Alam Ifrad  
**Institution**: Dickinson College  
**Supervisor**: Professor John MacCormick  
**Duration**: Spring 2026 (10 weeks)

## Project Goals

1. Build a robust n-digit addition model and characterize failure points
2. Train with explicit chain-of-though - measure resource efficiency
3. Provide scratch space for self-developed reasoning - identify resource savings
4. Compare resource trade-offs between approaches

## Repository Structure
```
transformer-reasoning-research/
├── data.py              # Data generation for addition problems
├── model.py             # Transformer architecture
├── train.py             # Training loop with WandB logging
├── evaluate.py          # Evaluation and testing
├── config.py            # Experiment configurations
├── notebooks/           # Jupyter notebooks for exploration
└── experiments/         # Experiment-specific scripts
```

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch wandb numpy matplotlib pandas jupyter
```

## Running Experiments
```bash
# Login to WandB
wandb login

# Train baseline model
python train.py
```


## References

- nanoGPT: https://github.com/karpathy/nanoGPT
- Scratchpad paper: [To be added]

## License

Academic research conducted at Dickinson College under the supervision of Professor John MacCormick. License to be determined.
