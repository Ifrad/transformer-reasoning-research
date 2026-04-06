import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm

from data_scratch import prepare_scratch_dataset, create_scratch_vocabulary
from model import AdditionTransformer


class ScratchDataset(Dataset):
    """PyTorch Dataset wrapper for scratch space data"""
    def __init__(self, encoded_data):
        self.data = encoded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.data[idx], dtype=torch.long)
        return tokens[:-1], tokens[1:]


def build_answer_mask(targets, end_s_idx):
    """Build mask that is True only for positions after [/S] in each sequence."""
    is_end_s = (targets == end_s_idx).int()
    eos_pos = is_end_s.argmax(dim=1)  # first [/S] position per row
    positions = torch.arange(targets.shape[1], device=targets.device).unsqueeze(0)
    return positions > eos_pos.unsqueeze(1)


def train_epoch(model, train_loader, optimizer, criterion, device, end_s_idx):
    """Train for one epoch. Loss is computed ONLY on answer digit positions (after [/S])."""
    model.train()
    total_loss = 0

    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss_per_token = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1)
        ).reshape(targets.shape)

        answer_mask = build_answer_mask(targets, end_s_idx)

        loss = (loss_per_token * answer_mask.float()).sum() / answer_mask.sum()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device, end_s_idx):
    """Evaluate model — accuracy on answer digits only (positions after [/S])."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss_per_token = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            ).reshape(targets.shape)

            answer_mask = build_answer_mask(targets, end_s_idx)

            loss = (loss_per_token * answer_mask.float()).sum() / answer_mask.sum()
            total_loss += loss.item()

            predictions = outputs.argmax(dim=-1)
            correct += ((predictions == targets) & answer_mask).sum().item()
            total += answer_mask.sum().item()

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--digits', type=int, default=2, choices=[2, 3])
    args = parser.parse_args()
    n_digits = args.digits

    config = {
        "n_digits": n_digits,
        "num_samples": 50000,
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 50,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 512,
        "mode": "scratch",
    }

    wandb.init(
        project="transformer-addition-scratch",
        config=config,
        name=f"scratch-{n_digits}digit"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Preparing scratch dataset...")
    train_data, val_data, char_to_idx, idx_to_char, vocab_size = prepare_scratch_dataset(
        n_digits=n_digits,
        num_samples=config["num_samples"]
    )
    config["vocab_size"] = vocab_size

    train_dataset = ScratchDataset(train_data)
    val_dataset = ScratchDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    print("Creating model...")
    model = AdditionTransformer(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        max_seq_len=64
    ).to(device)

    # Resolve [/S] token index for answer masking
    scratch_vocab = create_scratch_vocabulary()[0]
    end_s_idx = scratch_vocab['[/S]']

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vocab size: {vocab_size}")
    print(f"Sequence length (input): {len(train_data[0]) - 1}")
    print(f"Answer length: {n_digits + 1} digits")
    print(f"Loss masking: supervise only tokens after [/S] (idx {end_s_idx})")

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    print("Starting training...")
    best_accuracy = 0
    model_path = f"best_model_scratch_{n_digits}digit.pt"

    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, end_s_idx)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, end_s_idx)

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

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)

    print(f"\nBest validation accuracy: {best_accuracy:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
