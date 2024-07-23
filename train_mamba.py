import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import math
from data_preparation import load_and_preprocess_data, create_dataloaders
from mamba_model import create_mamba_model

def compute_metrics(logits, targets):
    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    perplexity = math.exp(loss.item())
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == targets).float().mean().item()
    return {
        'loss': loss,  # Return the tensor, not the item
        'loss_value': loss.item(),  # Add this for logging
        'perplexity': perplexity,
        'accuracy': accuracy
    }

def train(model, train_dataloader, val_dataloader, num_epochs, lr, device):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    wandb.init(project="mamba-next-word-prediction", config={
        "d_model": args.d_model,
        "n_layer": args.n_layer,
        "d_state": args.d_state,
        "d_conv": args.d_conv,
        "expand": args.expand,
        "learning_rate": args.lr,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
    })

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_metrics = {'train_loss': 0, 'train_perplexity': 0, 'train_accuracy': 0}
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            targets = input_ids.clone()
            targets[:, :-1] = input_ids[:, 1:]
            targets[:, -1] = input_ids[:, 0]

            optimizer.zero_grad()
            outputs = model(input_ids)
            metrics = compute_metrics(outputs, targets)
            loss = metrics['loss']  # This is now a tensor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1

            # Update epoch metrics
            epoch_metrics['train_loss'] += metrics['loss_value']
            epoch_metrics['train_perplexity'] += metrics['perplexity']
            epoch_metrics['train_accuracy'] += metrics['accuracy']

            if global_step % 50 == 0:
                wandb.log({
                    "train_loss": metrics['loss_value'],
                    "train_perplexity": metrics['perplexity'],
                    "train_accuracy": metrics['accuracy']
                }, step=global_step)

            if global_step % 250 == 0:
                val_metrics = evaluate(model, val_dataloader, device)
                wandb.log(val_metrics, step=global_step)
                model.train()  # Switch back to train mode after evaluation

        # Log epoch-level metrics
        epoch_metrics = {k: v / len(train_dataloader) for k, v in epoch_metrics.items()}
        wandb.log(epoch_metrics, step=global_step)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_metrics['train_loss']:.4f}, "
              f"Train Perplexity: {epoch_metrics['train_perplexity']:.4f}, "
              f"Train Accuracy: {epoch_metrics['train_accuracy']:.4f}")

        scheduler.step()

    wandb.finish()
    return model

def evaluate(model, dataloader, device):
    model.eval()
    total_metrics = {'val_loss': 0, 'val_perplexity': 0, 'val_accuracy': 0}
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = input_ids.clone()
            targets[:, :-1] = input_ids[:, 1:]
            targets[:, -1] = input_ids[:, 0]

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
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of Mamba layers")
    parser.add_argument("--d_state", type=int, default=16, help="SSM state expansion factor")
    parser.add_argument("--d_conv", type=int, default=4, help="Local convolution width")
    parser.add_argument("--expand", type=int, default=2, help="Block expansion factor")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
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
        expand=args.expand
    )

    print(f"Created Mamba model with parameters:")
    print(f"d_model: {args.d_model}")
    print(f"n_layer: {args.n_layer}")
    print(f"d_state: {args.d_state}")
    print(f"d_conv: {args.d_conv}")
    print(f"expand: {args.expand}")

    trained_model = train(model, train_dataloader, val_dataloader, args.num_epochs, args.lr, device)

    # Save the trained model
    torch.save(trained_model.state_dict(), "mamba_lm.pth")

# sample run: python train_mamba.py --d_model 512 --n_layer 8 --d_state 32 --d_conv 8 --expand 4 --num_epochs 20 --lr 5e-4 --batch_size 64