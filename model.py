import torch
import torch.nn as nn
import math

class AdditionTransformer(nn.Module):
    """
    Simple character-level transformer for addition
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, max_seq_len=20):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len) token indices
        Returns:
            (batch_size, seq_len, vocab_size) logits
        """
        # Embed and add positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transform
        x = self.transformer(x, mask=mask)
        
        # Project to vocabulary
        logits = self.fc_out(x)
        
        return logits

class PositionalEncoding(nn.Module):
    """Standard positional encoding"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:x.size(1), :]

# Test
if __name__ == "__main__":
    vocab_size = 12  # 0-9, +, =
    model = AdditionTransformer(vocab_size)
    
    # Test forward pass
    batch_size = 4
    seq_len = 10
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
