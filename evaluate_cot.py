import torch
import random
from data import create_vocabulary
from model import AdditionTransformer

def build_cot_string(a, b, n_digits):
    """Build the CoT reasoning string for a given addition problem."""
    a_str = f"{a:0{n_digits}d}"
    b_str = f"{b:0{n_digits}d}"
    result = a + b
    result_str = f"{result:0{n_digits+1}d}"

    steps = []
    carry = 0
    for pos in range(n_digits - 1, -1, -1):
        d_a = int(a_str[pos])
        d_b = int(b_str[pos])
        total = d_a + d_b + carry

        if pos == n_digits - 1:
            step_expr = f"{d_a}+{d_b}={total}"
        else:
            step_expr = f"{d_a}+{d_b}+{carry}={total}"

        if total >= 10:
            steps.append(f"{step_expr}(c1)")
            carry = 1
        else:
            steps.append(f"{step_expr}(nc)")
            carry = 0

    return f"{a_str}+{b_str}=" + ",".join(steps) + f",result={result_str}"

def generate_answer(model, cot_str, char_to_idx, idx_to_char, device, n_digits):
    """Feed the full CoT string and extract predicted answer digits after 'result='.

    Teacher-forcing: the model sees the entire reasoning chain (minus the last char)
    and we read its predictions for the answer positions after 'result='.
    """
    model.eval()

    input_str = cot_str[:-1]
    tokens = [char_to_idx[ch] for ch in input_str]
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predictions = output[0].argmax(dim=-1)

    # prediction[i] predicts the char at position i+1 in the full string
    result_eq_pos = input_str.index("result=") + len("result=") - 1
    answer_len = n_digits + 1
    answer_predictions = predictions[result_eq_pos : result_eq_pos + answer_len]

    return ''.join(idx_to_char[t.item()] for t in answer_predictions)

def requires_carry(a, b, n_digits):
    """Check if any digit position in the addition requires a carry"""
    carries = []
    carry = 0
    a_str = f"{a:0{n_digits}d}"
    b_str = f"{b:0{n_digits}d}"

    for i in range(n_digits - 1, -1, -1):
        digit_sum = int(a_str[i]) + int(b_str[i]) + carry
        carry = digit_sum // 10
        carries.append(carry > 0)

    return any(carries), carries

def analyze_errors(predicted, true_answer, n_digits):
    """
    Break down what kind of error was made.
    Positions are answer-relative: 0 = first digit of answer, 1 = second, etc.
    """
    answer_len = n_digits + 1
    pred_ans = (predicted + "X" * answer_len)[:answer_len]
    true_ans = (true_answer + "X" * answer_len)[:answer_len]

    analysis = {
        'correct': pred_ans == true_ans,
        'position_error': False,
        'digit_errors': [],
        'off_by_one': False,
    }

    for i in range(answer_len):
        p, t = pred_ans[i], true_ans[i]
        if p != t:
            analysis['digit_errors'].append({
                'position': i,
                'predicted_digit': p,
                'true_digit': t
            })

    if sorted(pred_ans) == sorted(true_ans) and pred_ans != true_ans:
        analysis['position_error'] = True

    try:
        if abs(int(predicted) - int(true_answer)) == 1:
            analysis['off_by_one'] = True
    except ValueError:
        pass

    return analysis

def evaluate_by_digit(model, char_to_idx, idx_to_char, device, n_digits, num_samples=500):
    """Full evaluation with carry, position, and digit-level analysis"""
    max_num = 10 ** n_digits - 1
    answer_len = n_digits + 1

    # Generate random (a, b) pairs and build both CoT and plain-format strings
    problems = []
    for _ in range(num_samples):
        a = random.randint(0, max_num)
        b = random.randint(0, max_num)
        true_answer = f"{a + b:0{answer_len}d}"
        cot_str = build_cot_string(a, b, n_digits)
        problems.append((a, b, true_answer, cot_str))

    # Sanity check
    print(f"\n  Sanity check (first 3 {n_digits}-digit problems, answer={answer_len} digits):")
    for a, b, true_answer, cot_str in problems[:3]:
        predicted = generate_answer(model, cot_str, char_to_idx, idx_to_char, device, n_digits)
        match = "✓" if predicted == true_answer else "✗"
        len_ok = "✓" if len(predicted) == answer_len else f"✗ WRONG LEN (got {len(predicted)}, expected {answer_len})"
        plain = f"{a:0{n_digits}d}+{b:0{n_digits}d}={true_answer}"
        print(f"    {plain}  =>  true: {true_answer}  predicted: {predicted}  {match}  len: {len_ok}")

    # Counters
    total = len(problems)
    correct = 0

    carry_total, carry_correct = 0, 0
    no_carry_total, no_carry_correct = 0, 0

    position_errors = 0
    off_by_one_errors = 0

    digit_position_errors = [0] * answer_len
    digit_position_total = [0] * answer_len

    sample_errors = []

    for a, b, true_answer, cot_str in problems:
        predicted = generate_answer(model, cot_str, char_to_idx, idx_to_char, device, n_digits)

        has_carry, _ = requires_carry(a, b, n_digits)
        error_analysis = analyze_errors(predicted, true_answer, n_digits)

        if error_analysis['correct']:
            correct += 1

        if has_carry:
            carry_total += 1
            if error_analysis['correct']:
                carry_correct += 1
        else:
            no_carry_total += 1
            if error_analysis['correct']:
                no_carry_correct += 1

        if error_analysis['position_error']:
            position_errors += 1

        if not error_analysis['correct'] and error_analysis['off_by_one']:
            off_by_one_errors += 1

        for pos_error in error_analysis['digit_errors']:
            pos = pos_error['position']
            if 0 <= pos < answer_len:
                digit_position_errors[pos] += 1
        for i in range(answer_len):
            digit_position_total[i] += 1

        if not error_analysis['correct'] and len(sample_errors) < 8:
            plain = f"{a:0{n_digits}d}+{b:0{n_digits}d}={true_answer}"
            sample_errors.append({
                'problem': plain,
                'predicted': predicted,
                'true': true_answer,
                'has_carry': has_carry,
                'position_error': error_analysis['position_error'],
                'off_by_one': error_analysis['off_by_one'],
                'digit_errors': error_analysis['digit_errors']
            })

    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total,
        'carry_accuracy': carry_correct / carry_total if carry_total > 0 else None,
        'no_carry_accuracy': no_carry_correct / no_carry_total if no_carry_total > 0 else None,
        'carry_total': carry_total,
        'no_carry_total': no_carry_total,
        'position_errors': position_errors,
        'off_by_one_errors': off_by_one_errors,
        'digit_position_errors': digit_position_errors,
        'digit_position_total': digit_position_total,
        'sample_errors': sample_errors
    }

def print_results(n_digits, results):
    print(f"\n{'='*50}")
    print(f"  {n_digits}-DIGIT COT ADDITION ANALYSIS")
    print(f"{'='*50}")

    print(f"\n OVERALL ACCURACY: {results['accuracy']*100:.1f}% ({results['correct']}/{results['total']})")

    print(f"\n CARRY ERROR ANALYSIS:")
    if results['carry_accuracy'] is not None:
        print(f"  Problems WITH carry:    {results['carry_accuracy']*100:.1f}% accurate  ({results['carry_total']} problems)")
        print(f"  Problems WITHOUT carry: {results['no_carry_accuracy']*100:.1f}% accurate  ({results['no_carry_total']} problems)")
        diff = (results['no_carry_accuracy'] - results['carry_accuracy']) * 100
        print(f"  Carry penalty: -{diff:.1f}% accuracy drop when carry is involved")

    print(f"\n DIGIT-BY-DIGIT ACCURACY (position 0 = first answer digit, 1 = second, etc.):")
    for i in range(len(results['digit_position_total'])):
        total_pos = results['digit_position_total'][i]
        if total_pos > 0:
            errors = results['digit_position_errors'][i]
            acc = (total_pos - errors) / total_pos * 100
            bar = '█' * int(acc / 5)
            print(f"  Position {i}: {acc:.1f}% correct  {bar}")

    print(f"\n OTHER ERROR PATTERNS:")
    print(f"  Position errors (right digits, wrong order): {results['position_errors']}")
    print(f"  Off-by-one errors: {results['off_by_one_errors']}")

    print(f"\n SAMPLE ERRORS:")
    for e in results['sample_errors'][:5]:
        carry_tag = "[CARRY]" if e['has_carry'] else "[NO CARRY]"
        pos_tag = "[POSITION ERROR]" if e['position_error'] else ""
        obo_tag = "[OFF BY ONE]" if e['off_by_one'] else ""
        print(f"  {e['problem']}")
        print(f"    Predicted: {e['predicted']}  True: {e['true']}  {carry_tag} {pos_tag} {obo_tag}")
        print()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    char_to_idx, idx_to_char, vocab_size, _ = create_vocabulary(cot=True)

    model = AdditionTransformer(vocab_size=vocab_size, max_seq_len=128)
    model.load_state_dict(torch.load("best_model_cot_3digit.pt", map_location=device))
    model.to(device)
    print(f"Model loaded successfully (vocab_size={vocab_size})")

    for n_digits in [3]:
        results = evaluate_by_digit(model, char_to_idx, idx_to_char, device, n_digits)
        print_results(n_digits, results)

if __name__ == "__main__":
    main()
