import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import wandb
import math
import argparse
from typing import Dict, Any

from transformers import get_cosine_schedule_with_warmup
from data_preparation import load_and_preprocess_data, create_dataloaders
from mamba_model import create_mamba_model

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    destroy_process_group()

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

def train(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    dataset, tokenizer = load_and_preprocess_data()
    train_sampler = DistributedSampler(dataset['train'], num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(dataset['validation'], num_replicas=world_size, rank=rank, shuffle=False)
    
    train_dataloader = DataLoader(dataset['train'], batch_size=args.batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(dataset['validation'], batch_size=args.batch_size, sampler=val_sampler)

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
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    if args.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=args.warmup_steps, 
                                                    num_training_steps=args.num_epochs * len(train_dataloader))
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    scaler = GradScaler()

    if rank == 0:
        wandb.init(project="mamba-next-word-prediction", config=vars(args))

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_metrics = {'train_loss': 0, 'train_perplexity': 0, 'train_accuracy': 0}
        
        for i, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            targets = input_ids.clone()
            targets[:, :-1] = input_ids[:, 1:]
            targets[:, -1] = input_ids[:, 0]

            with autocast():
                outputs = model(input_ids)
                metrics = compute_metrics(outputs, targets)
                loss = metrics['loss'] / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # Update epoch metrics
                epoch_metrics['train_loss'] += metrics['loss_value']
                epoch_metrics['train_perplexity'] += metrics['perplexity']
                epoch_metrics['train_accuracy'] += metrics['accuracy']

                if rank == 0 and i % 50 == 0:
                    wandb.log({
                        "train_loss": metrics['loss_value'],
                        "train_perplexity": metrics['perplexity'],
                        "train_accuracy": metrics['accuracy'],
                        "learning_rate": scheduler.get_last_lr()[0],
                    })

        # Evaluate on validation set
        val_metrics = evaluate(model, val_dataloader, device)
        
        if rank == 0:
            wandb.log(val_metrics)
            
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                torch.save(model.module.state_dict(), "best_model.pth")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

            # Log epoch-level metrics
            epoch_metrics = {k: v / len(train_dataloader) for k, v in epoch_metrics.items()}
            wandb.log(epoch_metrics)
            print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {epoch_metrics['train_loss']:.4f}, "
                  f"Train Perplexity: {epoch_metrics['train_perplexity']:.4f}, "
                  f"Train Accuracy: {epoch_metrics['train_accuracy']:.4f}")

    if rank == 0:
        wandb.finish()
    
    cleanup()

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
    return avg_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Mamba-Topoformer model for next word prediction")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_layer", type=int, default=8, help="Number of Mamba layers")
    parser.add_argument("--d_state", type=int, default=64, help="SSM state expansion factor")
    parser.add_argument("--d_conv", type=int, default=4, help="Local convolution width")
    parser.add_argument("--expand", type=int, default=2, help="Block expansion factor")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "linear"], help="Type of learning rate scheduler")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps for the scheduler")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use for training")
    args = parser.parse_args()

    torch.multiprocessing.spawn(train, args=(args.num_gpus, args), nprocs=args.num_gpus, join=True)

# sample run:
# python train_mamba.py --d_model 512 --n_layer 8 --d_state 64 --d_conv 4 --expand 2 --dropout 0.2 --num_epochs 30 --lr 1e-4 --scheduler cosine --warmup_steps 1000 --batch_size 32 --gradient_accumulation_steps 4 --max_grad_norm 1.0 --patience 5 --num_gpus 4