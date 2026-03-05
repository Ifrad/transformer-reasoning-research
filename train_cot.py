import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm

from data import prepare_dataset
from model import AdditionTransformer

class AdditionDataset(Dataset):
    """PyTorch Dataset wrapper"""
    def __init__(self, encoded_data):
        self.data = encoded_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = torch.tensor(self.data[idx], dtype=torch.long)
        # Input: everything except last token
        # Target: everything except first token (shifted by 1)
        return tokens[:-1], tokens[1:]

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Reshape for loss computation
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device, idx_to_char, pad_idx):
    """Evaluate model (accuracy excludes padding)"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
            
            # Calculate accuracy (exclude padding)
            predictions = outputs.argmax(dim=-1)
            mask = targets != pad_idx
            correct += ((predictions == targets) & mask).sum().item()
            total += mask.sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, accuracy

def main():
    # Configuration
    config = {
        "n_digits": 3,
        "num_samples": 10000,
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 50,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 512,
        "vocab_size": 24,
        "mode": "chain_of_thought",
    }
    
    # Initialize WandB
    wandb.init(
        project="transformer-addition-baseline",
        config=config,
        name="cot-3digit-addition"
    )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data (CoT format with expanded vocabulary and PAD)
    print("Preparing CoT dataset...")
    train_data, val_data, char_to_idx, idx_to_char, vocab_size, pad_idx = prepare_dataset(
        n_digits=config["n_digits"],
        num_samples=config["num_samples"],
        cot=True
    )
    
    def collate_fn(batch):
        """Pad sequences in batch to same length using pad_idx"""
        inputs, targets = zip(*batch)
        max_len = max(inp.size(0) for inp in inputs)
        padded_inputs = []
        padded_targets = []
        for inp, tgt in zip(inputs, targets):
            pad_len = max_len - inp.size(0)
            padded_inputs.append(
                torch.cat([inp, torch.full((pad_len,), pad_idx, dtype=inp.dtype)])
            )
            padded_targets.append(
                torch.cat([tgt, torch.full((pad_len,), pad_idx, dtype=tgt.dtype)])
            )
        return torch.stack(padded_inputs), torch.stack(padded_targets)
    
    train_dataset = AdditionDataset(train_data)
    val_dataset = AdditionDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        collate_fn=collate_fn
    )
    
    # Model (vocab_size=24 for CoT with PAD)
    print("Creating model...")
    model = AdditionTransformer(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        max_seq_len=128
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup (ignore padding in loss)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Training loop
    print("Starting training...")
    best_accuracy = 0
    
    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, idx_to_char, pad_idx)
        
        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model (separate file from baseline)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model_cot_3digit.pt")
            wandb.save("best_model_cot_3digit.pt")
    
    print(f"\nBest validation accuracy: {best_accuracy:.4f}")
    wandb.finish()

if __name__ == "__main__":
    main()
