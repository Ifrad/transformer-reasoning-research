import random

SCRATCH_LEN = 14

def create_scratch_vocabulary():
    """Extended vocabulary for scratch space experiments.
    Base tokens: 0123456789+= (indices 0-11, same as data.py)
    New tokens: [S] (idx 12), [/S] (idx 13), ~ (idx 14)
    Total: 15 tokens
    """
    base_chars = '0123456789+='
    char_to_idx = {ch: i for i, ch in enumerate(base_chars)}
    idx_to_char = {i: ch for i, ch in enumerate(base_chars)}

    char_to_idx['[S]'] = len(base_chars)
    char_to_idx['[/S]'] = len(base_chars) + 1
    char_to_idx['~'] = len(base_chars) + 2
    idx_to_char[len(base_chars)] = '[S]'
    idx_to_char[len(base_chars) + 1] = '[/S]'
    idx_to_char[len(base_chars) + 2] = '~'

    vocab_size = len(base_chars) + 3
    return char_to_idx, idx_to_char, vocab_size


def generate_scratch_dataset(num_samples, num_digits):
    """Generate addition problems with scratch space.

    Format: "12+34=[S]~~~~~~~~~~~~~~[/S]046"
      - num_digits=2: operands are 2-digit, answer is 3-digit (zero-padded)
      - num_digits=3: operands are 3-digit, answer is 4-digit (zero-padded)
      - Always 14 ~ tokens in the scratch space

    Args:
        num_samples: how many examples to generate
        num_digits: digits per operand (2 or 3)

    Returns:
        List of strings in scratch format
    """
    data = []
    max_num = 10 ** num_digits - 1
    scratch = '~' * SCRATCH_LEN

    for _ in range(num_samples):
        a = random.randint(0, max_num)
        b = random.randint(0, max_num)
        result = a + b
        problem = (
            f"{a:0{num_digits}d}+{b:0{num_digits}d}="
            f"[S]{scratch}[/S]"
            f"{result:0{num_digits+1}d}"
        )
        data.append(problem)

    return data


def build_scratch_string(a, b, n_digits):
    """Build the scratch format string for a specific (a, b) pair."""
    scratch = '~' * SCRATCH_LEN
    result = a + b
    return (
        f"{a:0{n_digits}d}+{b:0{n_digits}d}="
        f"[S]{scratch}[/S]"
        f"{result:0{n_digits+1}d}"
    )


def encode_scratch_data(data, char_to_idx):
    """Tokenize strings with multi-character token support for [S] and [/S]."""
    encoded = []
    for problem in data:
        tokens = []
        i = 0
        while i < len(problem):
            if problem[i:i+4] == '[/S]':
                tokens.append(char_to_idx['[/S]'])
                i += 4
            elif problem[i:i+3] == '[S]':
                tokens.append(char_to_idx['[S]'])
                i += 3
            else:
                tokens.append(char_to_idx[problem[i]])
                i += 1
        encoded.append(tokens)
    return encoded


def decode_scratch_tokens(tokens, idx_to_char):
    """Convert token indices back to string."""
    return ''.join(idx_to_char[t] for t in tokens)


def prepare_scratch_dataset(n_digits, num_samples, train_split=0.8):
    """Complete data preparation pipeline for scratch space experiments.

    Returns:
        train_data, val_data, char_to_idx, idx_to_char, vocab_size
    """
    data = generate_scratch_dataset(num_samples, n_digits)
    char_to_idx, idx_to_char, vocab_size = create_scratch_vocabulary()
    encoded = encode_scratch_data(data, char_to_idx)

    split_idx = int(len(encoded) * train_split)
    train_data = encoded[:split_idx]
    val_data = encoded[split_idx:]

    return train_data, val_data, char_to_idx, idx_to_char, vocab_size


if __name__ == "__main__":
    char_to_idx, idx_to_char, vocab_size = create_scratch_vocabulary()
    print(f"Scratch vocabulary size: {vocab_size}")
    print(f"Tokens: {char_to_idx}")

    for nd in [2, 3]:
        print(f"\n{nd}-digit scratch examples:")
        samples = generate_scratch_dataset(5, nd)
        for s in samples:
            print(f"  {s}")

        encoded = encode_scratch_data(samples, char_to_idx)
        decoded = [decode_scratch_tokens(e, idx_to_char) for e in encoded]
        assert samples == decoded, "Encoding roundtrip failed!"
        print(f"  Encoding roundtrip OK, seq_len={len(encoded[0])}")

    for nd in [2, 3]:
        data = generate_scratch_dataset(50000, nd)
        filename = f"data_scratch_{nd}digit.txt"
        with open(filename, 'w') as f:
            for line in data:
                f.write(line + '\n')
        print(f"\nSaved {len(data)} samples to {filename}")
