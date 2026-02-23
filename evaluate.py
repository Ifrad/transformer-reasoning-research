import torch
from data import generate_addition_pairs, create_vocabulary
from model import AdditionTransformer

def generate_answer(model, problem_str, char_to_idx, idx_to_char, device, n_digits):
    """Feed a problem like '12+34=046' and let the model complete it.

    Next-token predictor: prediction at position i predicts character i+1.
    Answer is always padded to n_digits+1 digits in training data.
    """
    model.eval()

    input_str = problem_str[:-1]
    tokens = [char_to_idx[ch] for ch in input_str]
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predictions = output[0].argmax(dim=-1)

    eq_pos = input_str.index('=')
    answer_len = n_digits + 1
    answer_predictions = predictions[eq_pos : eq_pos + answer_len]

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
    pred_ans = (predicted + "X" * answer_len)[:answer_len]  # pad/truncate to answer length
    true_ans = (true_answer + "X" * answer_len)[:answer_len]

    analysis = {
        'correct': pred_ans == true_ans,
        'position_error': False,   # right digits, wrong positions
        'digit_errors': [],        # which answer positions were wrong (0-indexed within answer)
        'off_by_one': False,       # predicted answer is ±1 from true
    }

    # Digit-by-digit comparison: position i = i-th digit of answer (answer-relative)
    for i in range(answer_len):
        p, t = pred_ans[i], true_ans[i]
        if p != t:
            analysis['digit_errors'].append({
                'position': i,  # answer-relative: 0 = first answer digit
                'predicted_digit': p,
                'true_digit': t
            })

    # Position error: sorted digits match but order is wrong
    if sorted(pred_ans) == sorted(true_ans) and pred_ans != true_ans:
        analysis['position_error'] = True

    # Off-by-one check (use unpadded strings for numeric comparison)
    try:
        if abs(int(predicted) - int(true_answer)) == 1:
            analysis['off_by_one'] = True
    except ValueError:
        pass

    return analysis

def evaluate_by_digit(model, char_to_idx, idx_to_char, device, n_digits, num_samples=500):
    """Full evaluation with carry, position, and digit-level analysis"""
    problems = generate_addition_pairs(n_digits, num_samples)

    # Sanity check: print first 3 problems with true vs predicted before full eval
    answer_len = n_digits + 1
    print(f"\n  Sanity check (first 3 {n_digits}-digit problems, answer={answer_len} digits):")
    for problem in problems[:3]:
        true_answer = problem.split('=')[1]
        predicted = generate_answer(model, problem, char_to_idx, idx_to_char, device, n_digits)
        match = "✓" if predicted == true_answer else "✗"
        len_ok = "✓" if len(predicted) == answer_len else f"✗ WRONG LEN (got {len(predicted)}, expected {answer_len})"
        print(f"    {problem}  =>  true: {true_answer}  predicted: {predicted}  {match}  len: {len_ok}")

    # Counters
    total = len(problems)
    correct = 0
    
    carry_total, carry_correct = 0, 0
    no_carry_total, no_carry_correct = 0, 0
    
    position_errors = 0
    off_by_one_errors = 0
    
    # Track errors per answer position: 0 = first answer digit, 1 = second, etc.
    digit_position_errors = [0] * (n_digits + 1)
    digit_position_total = [0] * (n_digits + 1)
    
    sample_errors = []
    
    for problem in problems:
        parts = problem.split('=')
        equation = parts[0]
        true_answer = parts[1]
        
        a, b = equation.split('+')
        a, b = int(a), int(b)
        
        predicted = generate_answer(model, problem, char_to_idx, idx_to_char, device, n_digits)
        
        has_carry, _ = requires_carry(a, b, n_digits)
        error_analysis = analyze_errors(predicted, true_answer, n_digits)
        
        # Overall accuracy
        if error_analysis['correct']:
            correct += 1
        
        # Carry vs no-carry accuracy
        if has_carry:
            carry_total += 1
            if error_analysis['correct']:
                carry_correct += 1
        else:
            no_carry_total += 1
            if error_analysis['correct']:
                no_carry_correct += 1
        
        # Position errors
        if error_analysis['position_error']:
            position_errors += 1
        
        # Off-by-one errors
        if not error_analysis['correct'] and error_analysis['off_by_one']:
            off_by_one_errors += 1
        
        # Digit-by-digit errors: position is answer-relative (0 = first answer digit)
        for pos_error in error_analysis['digit_errors']:
            pos = pos_error['position']
            if 0 <= pos < n_digits + 1:
                digit_position_errors[pos] += 1
        for i in range(n_digits + 1):
            digit_position_total[i] += 1
        
        # Collect sample errors
        if not error_analysis['correct'] and len(sample_errors) < 8:
            sample_errors.append({
                'problem': problem,
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

def debug_answer_slice(model, char_to_idx, idx_to_char, device):
    """Verbose debug: run on hardcoded '12+34=046' and print position-by-position output."""
    problem = "12+34=046"
    n_digits = 2

    input_str = problem[:-1]
    tokens = [char_to_idx[ch] for ch in input_str]
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predictions = output[0].argmax(dim=-1)

    eq_pos = input_str.index('=')
    answer_len = n_digits + 1
    answer_predictions = predictions[eq_pos : eq_pos + answer_len]
    predicted_answer = ''.join(idx_to_char[t.item()] for t in answer_predictions)
    true_answer = problem.split('=')[1]

    print("\n" + "=" * 60)
    print("  VERBOSE DEBUG: answer slice on '12+34=046'")
    print("=" * 60)
    print(f"\n  Full input string fed to model: '{input_str}'")
    print(f"\n  Position-by-position (i | input_char | predicted_char | expected_char[i+1]):")
    print("  " + "-" * 55)
    for i in range(len(input_str)):
        input_ch = input_str[i]
        pred_ch = idx_to_char[predictions[i].item()]
        expected_ch = problem[i + 1]  # prediction at i targets char at i+1 in full problem
        match = "✓" if pred_ch == expected_ch else "✗"
        print(f"    i={i}  |  '{input_ch}'  ->  pred='{pred_ch}'  expected='{expected_ch}'  {match}")
    print(f"\n  Index of '=' in input string: eq_pos = {eq_pos}")
    print(f"  Answer slice: predictions[{eq_pos} : {eq_pos + answer_len}]")
    print(f"  Slice produces chars at indices {eq_pos}-{eq_pos + answer_len - 1}:")
    for j, idx in enumerate(range(eq_pos, eq_pos + answer_len)):
        ch = idx_to_char[predictions[idx].item()]
        print(f"    index {idx} -> '{ch}' (answer digit position {j})")
    print(f"\n  True answer:   '{true_answer}'")
    print(f"  Predicted:     '{predicted_answer}'")
    print("=" * 60 + "\n")

def print_results(n_digits, results):
    print(f"\n{'='*50}")
    print(f"  {n_digits}-DIGIT ADDITION ANALYSIS")
    print(f"{'='*50}")
    
    # Overall accuracy
    print(f"\n OVERALL ACCURACY: {results['accuracy']*100:.1f}% ({results['correct']}/{results['total']})")
    
    # Carry analysis
    print(f"\n CARRY ERROR ANALYSIS:")
    if results['carry_accuracy'] is not None:
        print(f"  Problems WITH carry:    {results['carry_accuracy']*100:.1f}% accurate  ({results['carry_total']} problems)")
        print(f"  Problems WITHOUT carry: {results['no_carry_accuracy']*100:.1f}% accurate  ({results['no_carry_total']} problems)")
        diff = (results['no_carry_accuracy'] - results['carry_accuracy']) * 100
        print(f"  Carry penalty: -{diff:.1f}% accuracy drop when carry is involved")
    
    # Digit position analysis
    print(f"\n DIGIT-BY-DIGIT ACCURACY (position 0 = first answer digit, 1 = second, etc.):")
    for i in range(len(results['digit_position_total'])):
        total_pos = results['digit_position_total'][i]
        if total_pos > 0:
            errors = results['digit_position_errors'][i]
            acc = (total_pos - errors) / total_pos * 100
            bar = '█' * int(acc / 5)  # visual bar
            print(f"  Position {i}: {acc:.1f}% correct  {bar}")
    
    # Other error types
    print(f"\n OTHER ERROR PATTERNS:")
    print(f"  Position errors (right digits, wrong order): {results['position_errors']}")
    print(f"  Off-by-one errors: {results['off_by_one_errors']}")
    
    # Sample errors
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
    
    char_to_idx, idx_to_char, vocab_size = create_vocabulary()
    
    model = AdditionTransformer(vocab_size=vocab_size)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)
    print("Model loaded successfully")

    debug_answer_slice(model, char_to_idx, idx_to_char, device)

    for n_digits in [2, 3]:  # model trained on 2-digit; 4-digit exceeds sequence length
        results = evaluate_by_digit(model, char_to_idx, idx_to_char, device, n_digits)
        print_results(n_digits, results)

if __name__ == "__main__":
    main()
