import argparse
import random
import math
from collections import Counter, defaultdict
import torch

from data_scratch import (
    create_scratch_vocabulary, build_scratch_string,
    encode_scratch_data, decode_scratch_tokens, SCRATCH_LEN
)
from model import AdditionTransformer


def extract_predictions(model, scratch_str, char_to_idx, idx_to_char, device, n_digits):
    """Single forward pass: returns (predicted_answer, scratch_token_ids, correct)."""
    model.eval()

    encoded = encode_scratch_data([scratch_str], char_to_idx)[0]
    input_tokens = encoded[:-1]
    input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predictions = output[0].argmax(dim=-1)

    pred_tokens = [p.item() for p in predictions]

    answer_len = n_digits + 1
    answer_tokens = pred_tokens[-answer_len:]
    predicted_answer = ''.join(idx_to_char[t] for t in answer_tokens)

    scratch_start = 2 * n_digits + 2
    scratch_token_ids = pred_tokens[scratch_start:scratch_start + SCRATCH_LEN]

    return predicted_answer, scratch_token_ids


def collect_data(model, char_to_idx, idx_to_char, device, n_digits, num_samples=500):
    """Generate problems and collect scratch space + accuracy data."""
    max_num = 10 ** n_digits - 1
    records = []

    for _ in range(num_samples):
        a = random.randint(0, max_num)
        b = random.randint(0, max_num)
        answer_len = n_digits + 1
        true_answer = f"{a + b:0{answer_len}d}"
        scratch_str = build_scratch_string(a, b, n_digits)

        predicted_answer, scratch_token_ids = extract_predictions(
            model, scratch_str, char_to_idx, idx_to_char, device, n_digits
        )

        records.append({
            'a': a,
            'b': b,
            'true_answer': true_answer,
            'predicted_answer': predicted_answer,
            'correct': predicted_answer == true_answer,
            'ones_digit': int(true_answer[-1]),
            'scratch_token_ids': scratch_token_ids,
        })

    return records


def token_frequency_analysis(records, idx_to_char):
    """(a) What tokens appear most frequently across all scratch positions?"""
    counter = Counter()
    for r in records:
        counter.update(r['scratch_token_ids'])

    total = sum(counter.values())
    lines = ["TOKEN FREQUENCY DISTRIBUTION IN SCRATCH SPACE", "-" * 50]
    for token_id, count in counter.most_common():
        token_str = idx_to_char[token_id]
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        lines.append(f"  {token_str:>5s} (idx {token_id:2d}): {count:6d} ({pct:5.1f}%)  {bar}")
    return lines


def positional_analysis(records, idx_to_char):
    """(b) For each of 14 positions, what token appears most frequently?"""
    position_counters = [Counter() for _ in range(SCRATCH_LEN)]
    for r in records:
        for pos, token_id in enumerate(r['scratch_token_ids']):
            position_counters[pos][token_id] += 1

    lines = [
        "",
        "POSITIONAL ANALYSIS (most common token at each scratch position)",
        "-" * 50
    ]
    for pos, counter in enumerate(position_counters):
        top3 = counter.most_common(3)
        total = sum(counter.values())
        top_str = ", ".join(
            f"{idx_to_char[tid]:>5s}={cnt/total*100:.0f}%"
            for tid, cnt in top3
        )
        lines.append(f"  Position {pos:2d}: {top_str}")

    # Check if any position shows strong consistency (dominant token > 50%)
    structured_positions = []
    for pos, counter in enumerate(position_counters):
        total = sum(counter.values())
        top_id, top_count = counter.most_common(1)[0]
        if top_count / total > 0.5:
            structured_positions.append(
                f"pos {pos} ({idx_to_char[top_id]}: {top_count/total*100:.0f}%)"
            )
    if structured_positions:
        lines.append(f"\n  Structured positions (>50% dominant): {', '.join(structured_positions)}")
    else:
        lines.append("\n  No position has a single dominant token (>50%)")

    return lines


def ones_digit_correlation(records, idx_to_char):
    """(c) Do problems with the same ones digit share scratch patterns?"""
    groups = defaultdict(list)
    for r in records:
        groups[r['ones_digit']].append(r['scratch_token_ids'])

    lines = [
        "",
        "ONES DIGIT CORRELATION (scratch tokens grouped by ones digit of answer)",
        "-" * 50
    ]

    # For each ones digit, compute per-position most common scratch token
    for digit in range(10):
        if digit not in groups or len(groups[digit]) < 5:
            continue
        position_modes = []
        for pos in range(SCRATCH_LEN):
            counter = Counter(tokens[pos] for tokens in groups[digit])
            top_id, top_count = counter.most_common(1)[0]
            dominance = top_count / len(groups[digit])
            position_modes.append((idx_to_char[top_id], dominance))

        # Show only positions with >40% dominance for this ones digit
        strong = [
            f"p{pos}={tok}({dom*100:.0f}%)"
            for pos, (tok, dom) in enumerate(position_modes)
            if dom > 0.4
        ]
        lines.append(
            f"  ones={digit} (n={len(groups[digit]):3d}): "
            + (', '.join(strong) if strong else "no strong patterns")
        )

    # Overall correlation metric: for each position, compute conditional entropy H(scratch|ones)
    # Lower entropy = stronger correlation
    lines.append("")
    lines.append("  Per-position conditional entropy H(scratch_token | ones_digit):")
    for pos in range(SCRATCH_LEN):
        h_cond = 0.0
        total_n = len(records)
        for digit in range(10):
            if digit not in groups:
                continue
            n_d = len(groups[digit])
            p_d = n_d / total_n
            counter = Counter(tokens[pos] for tokens in groups[digit])
            h_given_d = 0.0
            for count in counter.values():
                p = count / n_d
                if p > 0:
                    h_given_d -= p * math.log2(p)
            h_cond += p_d * h_given_d
        lines.append(f"    Position {pos:2d}: H = {h_cond:.3f} bits")

    return lines


def entropy_accuracy_split(records, idx_to_char):
    """(d) Split by scratch space entropy — is structured scratch more accurate?"""
    for r in records:
        counter = Counter(r['scratch_token_ids'])
        total = SCRATCH_LEN
        h = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                h -= p * math.log2(p)
        r['scratch_entropy'] = h

    entropies = [r['scratch_entropy'] for r in records]
    median_entropy = sorted(entropies)[len(entropies) // 2]

    low_entropy = [r for r in records if r['scratch_entropy'] <= median_entropy]
    high_entropy = [r for r in records if r['scratch_entropy'] > median_entropy]

    low_acc = sum(r['correct'] for r in low_entropy) / len(low_entropy) if low_entropy else 0
    high_acc = sum(r['correct'] for r in high_entropy) / len(high_entropy) if high_entropy else 0

    overall_acc = sum(r['correct'] for r in records) / len(records)

    lines = [
        "",
        "ENTROPY-BASED ACCURACY SPLIT",
        "-" * 50,
        f"  Median scratch entropy: {median_entropy:.3f} bits",
        f"  Low entropy  (structured, n={len(low_entropy):3d}): {low_acc*100:.1f}% accuracy",
        f"  High entropy (noisy,      n={len(high_entropy):3d}): {high_acc*100:.1f}% accuracy",
        f"  Overall accuracy: {overall_acc*100:.1f}%",
        f"  Difference: {(low_acc - high_acc)*100:+.1f}% (positive = structured is better)",
    ]

    # Show entropy distribution
    lines.append("")
    lines.append("  Entropy distribution:")
    bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    for i in range(len(bins) - 1):
        count = sum(1 for e in entropies if bins[i] <= e < bins[i+1])
        bar = '█' * (count // max(1, len(records) // 50))
        lines.append(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}): {count:4d}  {bar}")

    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--digits', type=int, default=2, choices=[2, 3])
    parser.add_argument('--samples', type=int, default=500)
    args = parser.parse_args()
    n_digits = args.digits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    char_to_idx, idx_to_char, vocab_size = create_scratch_vocabulary()

    model_path = f"best_model_scratch_{n_digits}digit.pt"
    model = AdditionTransformer(vocab_size=vocab_size, max_seq_len=64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"Model loaded from {model_path}")

    print(f"Collecting data from {args.samples} problems...")
    records = collect_data(model, char_to_idx, idx_to_char, device, n_digits, args.samples)
    accuracy = sum(r['correct'] for r in records) / len(records)
    print(f"Overall accuracy: {accuracy*100:.1f}%\n")

    report_lines = [
        f"SCRATCH SPACE ANALYSIS REPORT ({n_digits}-digit addition)",
        f"Samples: {args.samples}",
        f"Overall accuracy: {accuracy*100:.1f}%",
        "=" * 60,
        "",
    ]

    report_lines += token_frequency_analysis(records, idx_to_char)
    report_lines += positional_analysis(records, idx_to_char)
    report_lines += ones_digit_correlation(records, idx_to_char)
    report_lines += entropy_accuracy_split(records, idx_to_char)

    report = '\n'.join(report_lines)
    print(report)

    report_path = "scratch_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report + '\n')
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
