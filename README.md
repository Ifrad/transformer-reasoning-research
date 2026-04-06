# Transformer Reasoning Research

**Research Project**: Investigating emergent reasoning capabilities in transformer models through chain-of-thought prompting and autonomous scratch space utilization.

**Researcher**: Mubashshir Alam Ifrad  
**Institution**: Dickinson College  
**Supervisor**: Professor John MacCormick  

## Project Goals

1. Build a robust n-digit addition model and characterize failure points
2. Train with explicit chain-of-though - measure resource efficiency
3. Provide scratch space for self-developed reasoning - identify resource savings
4. Compare resource trade-offs between approaches

## Repository Structure
```
transformer-reasoning-research/
├── data.py              # Baseline + CoT data generation
├── model.py             # Transformer architecture (AdditionTransformer)
├── train.py             # Baseline training loop
├── evaluate.py          # Baseline evaluation
├── train_cot.py         # Chain-of-thought training (Phase 2–3)
├── evaluate_cot.py      # CoT evaluation
├── data_scratch.py      # Scratch space data generation (Phase 4)
├── train_scratch.py     # Scratch space training with loss masking
├── evaluate_scratch.py  # Scratch space evaluation
└── analyze_scratch.py   # Scratch space pattern analysis
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


## Phase 4: Scratch Space Experiment

**Goal**: Train a character-level transformer on addition problems with an unsupervised
scratch space. The model is free to write anything in the scratch space — after training,
we analyze whether it spontaneously developed carry-based reasoning.

### Format

```
12+34=[S]~~~~~~~~~~~~~~[/S]046
123+456=[S]~~~~~~~~~~~~~~[/S]0579
```

- `[S]` and `[/S]` are special delimiter tokens marking the scratch region
- 14 `~` (mask) tokens fill the scratch space — the model's predictions here are **not supervised**
- Loss is computed **only on the answer digits** after `[/S]`
- Answer uses the same zero-padded format as the baseline (n_digits+1 digits)

### Vocabulary

Base tokens `0123456789+=` (indices 0–11) plus 3 new tokens:
| Token | Index | Purpose |
|-------|-------|---------|
| `[S]` | 12 | Scratch start delimiter |
| `[/S]` | 13 | Scratch end delimiter |
| `~` | 14 | Mask/scratch placeholder |

Total vocabulary size: **15**

### Files

| File | Description |
|------|-------------|
| `data_scratch.py` | Data generation with scratch format, vocabulary, encode/decode |
| `train_scratch.py` | Training loop with answer-only loss masking |
| `evaluate_scratch.py` | Accuracy evaluation with full sequence output |
| `analyze_scratch.py` | Novel analysis of scratch space patterns |

### Running

```bash
# Generate datasets (saved as data_scratch_2digit.txt / data_scratch_3digit.txt)
python data_scratch.py

# Train (--digits 2 or 3)
python train_scratch.py --digits 2
python train_scratch.py --digits 3

# Evaluate (prints 20 examples with scratch content + accuracy report)
python evaluate_scratch.py --digits 2

# Analyze scratch space patterns (saves scratch_analysis_report.txt)
python analyze_scratch.py --digits 2
```

### Analysis

`analyze_scratch.py` examines the model's unsupervised scratch space for:
1. **Token frequency** — which tokens does the model prefer to "write"?
2. **Positional structure** — do specific positions consistently use specific tokens?
3. **Ones-digit correlation** — do problems with the same ones digit produce similar scratch patterns?
4. **Entropy–accuracy split** — does structured (low-entropy) scratch content correlate with higher accuracy?

## References

- nanoGPT: https://github.com/karpathy/nanoGPT
- Scratchpad paper: [To be added]

## License

Academic research conducted at Dickinson College under the supervision of Professor John MacCormick. License to be determined.
