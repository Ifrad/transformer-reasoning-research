# Transformer Reasoning Research

Investigating whether small character-level transformers can develop reasoning strategies for arithmetic on their own, or whether explicit step-by-step structure is required for systematic generalization.

**Researcher**: Mubashshir Alam Ifrad | Dickinson College
**Supervisor**: Professor John MacCormick

---

## Summary

A character-level transformer (~400K parameters, 2 layers, 4 attention heads, 128-d embeddings) was trained on n-digit addition across four phases: a baseline, two chain-of-thought variants, and a scratch-space model given unsupervised free tokens between the problem and the answer.

| Phase | Format | 2-digit | 3-digit |
|---|---|---|---|
| 1 — Baseline | `12+34=046` | 9.6% | 0.2% |
| 2 — CoT | with carry steps | 98.8% | 84.8% |
| 3 — CoT + `ones=` | with ones reminder | — | **100.0%** |
| 4 — Scratch space | `[S]~~~~~~~~~~~~~~[/S]` | 98.87% | 11.2% |

All accuracies are problem-level (every digit must be correct).

**Central finding**: unstructured scratch space does not induce systematic reasoning. The model uses scratch tokens for statistical compression at small scale and collapses to single-bit heuristics at larger scale. Explicit structured chain-of-thought was the only approach that achieved systematic generalization.

---

## Repository Structure

```
transformer-reasoning-research/
├── data.py                 # Baseline + CoT data generation
├── model.py                # AdditionTransformer architecture
├── train.py                # Baseline training
├── evaluate.py             # Baseline evaluation
├── train_cot.py            # Chain-of-thought training
├── evaluate_cot.py         # CoT evaluation
├── data_scratch.py         # Scratch space data (Phase 4)
├── train_scratch.py        # Scratch training + loss masking
├── evaluate_scratch.py     # Scratch evaluation
└── analyze_scratch.py      # Scratch pattern analysis
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch wandb numpy matplotlib pandas jupyter
wandb login
```

---

## Phase 1 — Baseline

Trained on 10,000 two-digit problems in the format `12+34=046`. The training loop reported 88.9% token accuracy, but a custom evaluator measuring problem-level accuracy (all digits must be correct) revealed real performance: **9.6% on 2-digit, 0.2% on 3-digit**.

Root cause: the model outputs digits left to right, but carry propagation works right to left. It never learned to carry — it was essentially guessing the final digit.

```bash
python train.py
python evaluate.py
```

---

## Phase 2–3 — Chain-of-Thought

### Phase 2: explicit carry steps

Training data was redesigned with right-to-left carry reasoning between the problem and the answer:

```
47+68=7+8=15(c1),4+6+1=11(c1),result=115
```

Vocabulary expanded from 12 to 25 tokens. Accuracy reached **98.8% on 2-digit** and **84.8% on 3-digit**.

### Phase 3: the `ones=` reminder fix

The 3-digit model failed almost exclusively on the ones digit of the answer. Root cause: the ones digit is computed first in the reasoning chain but written last in `result=` — too far back in context. A failed attempt with 50,000 samples and 100 epochs overfit and dropped accuracy to 63.8%.

Fix: an explicit `ones=` reminder token inserted immediately before `result=`:

```
47+68=7+8=15(c1),4+6+1=11(c1),ones=5,result=115
```

Result: **100% problem-level accuracy on 3-digit addition** — zero errors.

```bash
python train_cot.py --digits 2
python train_cot.py --digits 3
python evaluate_cot.py --digits 3
```

---

## Phase 4 — Scratch Space

**Research question**: given free tokens between the problem and the answer, with no instruction on how to use them, does the model spontaneously discover carry-based reasoning?

### Format

```
12+34=[S]~~~~~~~~~~~~~~[/S]046
123+456=[S]~~~~~~~~~~~~~~[/S]0579
```

The 14 `~` tokens form a free scratch region. The model's predictions on those tokens are **not supervised** — loss is computed only on the answer digits after `[/S]`. The `[/S]` delimiter forces the model to commit to scratch content before seeing where the answer goes, preventing it from ignoring the scratch space entirely.

### Vocabulary

Base tokens `0123456789+=` (indices 0–11) plus three new tokens:

| Token | Index | Purpose |
|---|---|---|
| `[S]` | 12 | Scratch start delimiter |
| `[/S]` | 13 | Scratch end delimiter |
| `~` | 14 | Mask / scratch placeholder |

Total vocabulary size: **15**.

### Running

```bash
python data_scratch.py                   # generates data_scratch_2digit.txt and data_scratch_3digit.txt
python train_scratch.py --digits 2
python train_scratch.py --digits 3
python evaluate_scratch.py --digits 2    # prints 20 examples with scratch content
python analyze_scratch.py --digits 2     # saves scratch_analysis_report.txt
```

### Findings

**2-digit (98.87%)** — matches CoT accuracy, but analysis revealed no carry-based reasoning. The scratch space was dominated by digits 0 and 1 (37.5% and 19.0% of all scratch tokens) — the most statistically frequent digits in 2-digit addition answers. The model learned to cache answer statistics into the scratch tokens rather than compute intermediate steps. Near-zero entropy across 85.6% of sequences confirmed this was a consistent compression strategy, not emergent reasoning.

**3-digit (11.2%)** — the shortcut collapsed. With a much larger answer space (0–1998), statistical compression no longer worked. The model instead learned to encode a single bit of information about the ones digit of the answer, alternating between 0 and 1 across all 14 scratch positions depending on which ones-digit group the answer belonged to. Conditional entropy H(scratch | ones digit) dropped to roughly 1.0 bits — genuine structure, but only one bit of information, far too coarse to reconstruct full 3-digit answers.

### What `analyze_scratch.py` examines

1. **Token frequency** — which tokens does the model prefer to write?
2. **Positional structure** — do specific positions consistently use specific tokens?
3. **Ones-digit correlation** — do problems sharing a ones digit produce similar scratch patterns?
4. **Entropy–accuracy split** — does structured (low-entropy) scratch content correlate with higher accuracy?

---

## Conclusion

Unstructured scratch space does not induce systematic reasoning in small transformers. On small-scale problems it enables statistical memorization that superficially matches the performance of chain-of-thought, but that strategy does not scale. On larger problems it collapses to single-bit heuristics. Explicit, structured chain-of-thought — where intermediate reasoning steps are supervised token by token — was the only approach that achieved systematic generalization.

The finding has a practical implication for the broader question of emergent reasoning in language models: the structure of the intermediate tokens matters more than the mere presence of intermediate space. Giving a model room to think is not the same as teaching it how to think.

---

## References

- Nye, M., Andreassen, A. J., Gur-Ari, G., et al. (2021). *Show Your Work: Scratchpads for Intermediate Computation with Language Models.* arXiv:2112.00114.
- Karpathy, A. *nanoGPT.* https://github.com/karpathy/nanoGPT

---

## License

Academic research conducted at Dickinson College under the supervision of Professor John MacCormick. License to be determined.
