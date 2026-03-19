import random
import torch

def generate_addition_pairs(n_digits, num_samples):
    """
    Generate addition problems in format: "12+34=46"
    
    Args:
        n_digits: number of digits for each operand
        num_samples: how many examples to generate
    
    Returns:
        List of strings like ["12+34=46", "05+17=22", ...]
    """
    data = []
    max_num = 10 ** n_digits - 1
    
    for _ in range(num_samples):
        a = random.randint(0, max_num)
        b = random.randint(0, max_num)
        result = a + b
        
        # Format with leading zeros to keep consistent length
        problem = f"{a:0{n_digits}d}+{b:0{n_digits}d}={result:0{n_digits+1}d}"
        data.append(problem)
    
    return data

def generate_cot_addition_pairs(n_digits, num_samples):
    """
    Generate addition problems with explicit carry chain-of-thought reasoning.
    Format examples:
      - 12+34=046: "12+34=2+4=6(nc),1+3+0=4(nc),ones=6,result=046"
      - 47+68=115: "47+68=7+8=15(c1),4+6+1=11(c1),ones=5,result=115"
    
    Works right-to-left. At each position: digit_a + digit_b + carry_in.
    (c1) = result >= 10, carry 1 to next position, digit = result % 10
    (nc) = result < 10, no carry
    
    Args:
        n_digits: number of digits for each operand
        num_samples: how many examples to generate
    
    Returns:
        List of strings with COT reasoning
    """
    data = []
    max_num = 10 ** n_digits - 1
    
    for _ in range(num_samples):
        a = random.randint(0, max_num)
        b = random.randint(0, max_num)
        result = a + b
        
        a_str = f"{a:0{n_digits}d}"
        b_str = f"{b:0{n_digits}d}"
        result_str = f"{result:0{n_digits+1}d}"
        
        steps = []
        carry = 0
        for pos in range(n_digits - 1, -1, -1):  # right to left
            d_a = int(a_str[pos])
            d_b = int(b_str[pos])
            total = d_a + d_b + carry
            
            if pos == n_digits - 1:
                # Rightmost position: no carry_in in the display
                step_expr = f"{d_a}+{d_b}={total}"
            else:
                step_expr = f"{d_a}+{d_b}+{carry}={total}"
            
            if total >= 10:
                steps.append(f"{step_expr}(c1)")
                carry = 1
            else:
                steps.append(f"{step_expr}(nc)")
                carry = 0
        
        ones_digit = result_str[-1]
        cot = f"{a_str}+{b_str}=" + ",".join(steps) + f",ones={ones_digit},result={result_str}"
        data.append(cot)
    
    return data

def create_vocabulary(cot=False):
    """
    Create character-level vocabulary.
    When cot=False: base chars 0123456789+=
    When cot=True: extends with (, ), n, c, , o,r,e,s,u,l,t and P (PAD), vocab size 25
    """
    chars = '0123456789+='
    pad_idx = None
    if cot:
        chars += '(),ncoresultP'  # CoT markers; ones; result; P = PAD
        pad_idx = len(chars) - 1  # P is last
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char, len(chars), pad_idx

def encode_data(data, char_to_idx):
    """Convert string data to token indices"""
    encoded = []
    for problem in data:
        tokens = [char_to_idx[ch] for ch in problem]
        encoded.append(tokens)
    return encoded

def prepare_dataset(n_digits, num_samples, train_split=0.8, cot=False):
    """
    Complete data preparation pipeline
    
    Args:
        cot: if True, use CoT format data and extended vocabulary
    
    Returns:
        train_data, val_data, char_to_idx, idx_to_char, vocab_size
    """
    # Generate problems
    data = generate_cot_addition_pairs(n_digits, num_samples) if cot else generate_addition_pairs(n_digits, num_samples)
    
    # Create vocabulary
    char_to_idx, idx_to_char, vocab_size, pad_idx = create_vocabulary(cot=cot)
    
    # Encode
    encoded = encode_data(data, char_to_idx)
    
    # Split train/val
    split_idx = int(len(encoded) * train_split)
    train_data = encoded[:split_idx]
    val_data = encoded[split_idx:]
    
    return train_data, val_data, char_to_idx, idx_to_char, vocab_size, pad_idx

# Quick test
if __name__ == "__main__":
    # Test the data generation
    problems = generate_addition_pairs(n_digits=2, num_samples=5)
    print("Sample problems:")
    for p in problems:
        print(f"  {p}")
    
    # Test COT addition pairs
    print("\n2-digit COT addition (5 examples):")
    cot_2 = generate_cot_addition_pairs(n_digits=2, num_samples=5)
    for p in cot_2:
        print(f"  {p}")
    
    print("\n3-digit COT addition (5 examples):")
    cot_3 = generate_cot_addition_pairs(n_digits=3, num_samples=5)
    for p in cot_3:
        print(f"  {p}")
    
    # Test full pipeline (base vocabulary)
    train, val, c2i, i2c, vocab_size, _ = prepare_dataset(n_digits=2, num_samples=100)
    print(f"\nBase vocabulary size (for transformer): {vocab_size}")
    print(f"Training samples: {len(train)}")
    print(f"Validation samples: {len(val)}")
    print(f"First encoded example: {train[0]}")
    
    # Test CoT vocabulary and pipeline
    c2i_cot, i2c_cot, vocab_size_cot, pad_idx_cot = create_vocabulary(cot=True)
    print(f"\nCoT vocabulary size (for transformer): {vocab_size_cot}")
    print(f"CoT pad_idx: {pad_idx_cot}")
    print("CoT vocabulary chars:", "".join(i2c_cot[i] for i in sorted(i2c_cot.keys())))
    train_cot, val_cot, _, _, _, _ = prepare_dataset(n_digits=2, num_samples=100, cot=True)
    print(f"CoT training samples: {len(train_cot)}, validation: {len(val_cot)}")
