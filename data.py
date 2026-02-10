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

def create_vocabulary():
    """Create character-level vocabulary"""
    chars = '0123456789+='
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char, len(chars)

def encode_data(data, char_to_idx):
    """Convert string data to token indices"""
    encoded = []
    for problem in data:
        tokens = [char_to_idx[ch] for ch in problem]
        encoded.append(tokens)
    return encoded

def prepare_dataset(n_digits, num_samples, train_split=0.8):
    """
    Complete data preparation pipeline
    
    Returns:
        train_data, val_data, char_to_idx, idx_
to_char, vocab_size
    """
    # Generate problems
    data = generate_addition_pairs(n_digits, num_samples)
    
    # Create vocabulary
    char_to_idx, idx_to_char, vocab_size = create_vocabulary()
    
    # Encode
    encoded = encode_data(data, char_to_idx)
    
    # Split train/val
    split_idx = int(len(encoded) * train_split)
    train_data = encoded[:split_idx]
    val_data = encoded[split_idx:]
    
    return train_data, val_data, char_to_idx, idx_to_char, vocab_size

# Quick test
if __name__ == "__main__":
    # Test the data generation
    problems = generate_addition_pairs(n_digits=2, num_samples=5)
    print("Sample problems:")
    for p in problems:
        print(f"  {p}")
    
    # Test full pipeline
    train, val, c2i, i2c, vocab_size = prepare_dataset(n_digits=2, num_samples=100)
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Training samples: {len(train)}")
    print(f"Validation samples: {len(val)}")
    print(f"First encoded example: {train[0]}")
