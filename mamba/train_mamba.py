import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import math
import argparse
from typing import Dict, Any

from transformers import get_cosine_schedule_with_warmup
from data_preparation import load_and_preprocess_data, create_dataloaders
from mamba_model import create_mamba_model

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    perplexity = math.exp(loss.item())
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == targets).float().mean().item()
    return {
        'loss': loss,
        'loss_value': loss.item(),
        'perplexity': perplexity,
        'accuracy': accuracy
    }

def train(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, 
          num_epochs: int, lr: float, device: torch.device, 
          gradient_accumulation_steps: int, max_grad_norm: float,
          scheduler_type: str, warmup_steps: int):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    if scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=warmup_steps, 
                                                    num_training_steps=num_epochs * len(train_dataloader))
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    scaler = GradScaler()

    wandb.init(project="mamba-next-word-prediction", config={
        "d_model": args.d_model,
        "n_layer": args.n_layer,
        "d_state": args.d_state,
        "d_conv": args.d_conv,
        "expand": args.expand,
        "dropout": args.dropout,
        "learning_rate": args.lr,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "scheduler_type": args.scheduler,
        "warmup_steps": args.warmup_steps,
    })

    best_val_loss = float('inf')
    patience = args.patience
    patience_counter = 0

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_metrics = {'train_loss': 0, 'train_perplexity': 0, 'train_accuracy': 0}
        
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            targets = input_ids.clone()
            targets[:, :-1] = input_ids[:, 1:]
            targets[:, -1] = input_ids[:, 0]

            with autocast():
                outputs = model(input_ids)
                metrics = compute_metrics(outputs, targets)
                loss = metrics['loss'] / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Update epoch metrics
                epoch_metrics['train_loss'] += metrics['loss_value']
                epoch_metrics['train_perplexity'] += metrics['perplexity']
                epoch_metrics['train_accuracy'] += metrics['accuracy']

                if global_step % 50 == 0:
                    wandb.log({
                        "train_loss": metrics['loss_value'],
                        "train_perplexity": metrics['perplexity'],
                        "train_accuracy": metrics['accuracy'],
                        "learning_rate": scheduler.get_last_lr()[0],
                    }, step=global_step)

                if global_step % 250 == 0:
                    val_metrics = evaluate(model, val_dataloader, device)
                    wandb.log(val_metrics, step=global_step)
                    
                    if val_metrics['val_loss'] < best_val_loss:
                        best_val_loss = val_metrics['val_loss']
                        torch.save(model.state_dict(), "best_model.pth")
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        wandb.finish()
                        return model
                    
                    model.train()

            progress_bar.update(1)
        progress_bar.close()

        # Log epoch-level metrics
        epoch_metrics = {k: v / len(train_dataloader) for k, v in epoch_metrics.items()}
        wandb.log(epoch_metrics, step=global_step)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_metrics['train_loss']:.4f}, "
              f"Train Perplexity: {epoch_metrics['train_perplexity']:.4f}, "
              f"Train Accuracy: {epoch_metrics['train_accuracy']:.4f}")

    wandb.finish()
    return model

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_metrics = {'val_loss': 0, 'val_perplexity': 0, 'val_accuracy': 0}
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = input_ids.clone()
            targets[:, :-1] = input_ids[:, 1:]
            targets[:, -1] = input_ids[:, 0]

            with autocast():
                outputs = model(input_ids)
                metrics = compute_metrics(outputs, targets)
            
            total_metrics['val_loss'] += metrics['loss_value']
            total_metrics['val_perplexity'] += metrics['perplexity']
            total_metrics['val_accuracy'] += metrics['accuracy']

    avg_metrics = {k: v / len(dataloader) for k, v in total_metrics.items()}
    print(f"Validation Loss: {avg_metrics['val_loss']:.4f}, "
          f"Validation Perplexity: {avg_metrics['val_perplexity']:.4f}, "
          f"Validation Accuracy: {avg_metrics['val_accuracy']:.4f}")
    return avg_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Mamba model for next word prediction")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_layer", type=int, default=8, help="Number of Mamba layers")
    parser.add_argument("--d_state", type=int, default=64, help="SSM state expansion factor")
    parser.add_argument("--d_conv", type=int, default=4, help="Local convolution width")
    parser.add_argument("--expand", type=int, default=2, help="Block expansion factor")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "linear"], help="Type of learning rate scheduler")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps for the scheduler")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset, tokenizer = load_and_preprocess_data()
    train_dataloader, val_dataloader, _ = create_dataloaders(dataset, batch_size=args.batch_size)

    vocab_size = len(tokenizer)
    model = create_mamba_model(
        vocab_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout
    )

    print(f"Created Mamba model with parameters:")
    print(f"d_model: {args.d_model}")
    print(f"n_layer: {args.n_layer}")
    print(f"d_state: {args.d_state}")
    print(f"d_conv: {args.d_conv}")
    print(f"expand: {args.expand}")
    print(f"dropout: {args.dropout}")

    trained_model = train(model, train_dataloader, val_dataloader, args.num_epochs, args.lr, device, 
                          args.gradient_accumulation_steps, args.max_grad_norm,
                          args.scheduler, args.warmup_steps)

    # Save the final trained model
    torch.save(trained_model.state_dict(), "mamba_lm.pth")

# sample run:
# python train_mamba.py --d_model 512 --n_layer 8 --d_state 64 --d_conv 4 --expand 2 --dropout 0.2 --num_epochs 30 --lr 1e-4 --scheduler cosine --warmup_steps 1000 --batch_size 32 --gradient_accumulation_steps 4 --max_grad_norm 1.0 --patience 5