import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import wandb
from tqdm import tqdm
import requests
import os

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        
        self.in_proj = nn.Linear(d_model, self.d_inner)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv-1, groups=self.d_inner)
        
        self.x_proj = nn.Linear(self.d_inner, self.d_state)
        self.dt_proj = nn.Linear(self.d_inner, self.d_state)
        
        self.A = nn.Parameter(torch.randn(self.d_state))
        self.D = nn.Parameter(torch.randn(self.d_state, self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model)
    
    def forward(self, x):
        B, L, _ = x.shape
        
        x = self.in_proj(x)
        x = x.transpose(1, 2)
        x = self.conv(x)[:, :, :L]
        x = x.transpose(1, 2)
        
        x_ssm = self.x_proj(x)
        dt = F.softplus(self.dt_proj(x))
        
        A = -torch.exp(self.A).view(1, 1, -1)
        D = self.D
        
        dA = torch.exp(A * dt)
        dB = (1 - dA) / A
        
        z = torch.zeros(B, self.d_state, device=x.device)
        output = []
        for i in range(L):
            z = dA[:, i] * z + dB[:, i] * x_ssm[:, i]
            output.append(z)
        z = torch.stack(output, dim=1)
        
        y = torch.einsum('bls,si->bli', z, D)
        
        y = self.out_proj(y)
        
        return y

class MambaLM(nn.Module):
    def __init__(self, vocab_size, d_model, d_state, d_conv, expand, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        
        for layer in self.layers:
            x = x + layer(x)
        
        x = self.norm(x)
        x = self.lm_head(x)
        
        return x

def load_ptb_data(batch_size):
    path = 'ptb.train.txt'
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt"
        print(f"Downloading dataset from {url}")
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
            print("Download completed.")
        else:
            raise Exception(f"Failed to download the dataset. Status code: {response.status_code}")
    
    with open(path, 'r', encoding='utf-8') as f:
        raw_data = f.read().split()
    
    vocab = ['<unk>', '<eos>'] + sorted(set(raw_data))
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}
    
    data = [word_to_idx[word] for word in raw_data]
    
    train_data = data[:int(0.8*len(data))]
    val_data = data[int(0.8*len(data)):int(0.9*len(data))]
    test_data = data[int(0.9*len(data)):]
    
    return train_data, val_data, test_data, len(vocab), idx_to_word

class TextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index:index+self.seq_length]),
            torch.tensor(self.data[index+1:index+self.seq_length+1])
        )

def get_data_loaders(train_data, val_data, test_data, batch_size, seq_length):
    train_dataset = TextDataset(train_data, seq_length)
    val_dataset = TextDataset(val_data, seq_length)
    test_dataset = TextDataset(test_data, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                        desc=f"Epoch {epoch}", ncols=100)
    
    for batch, (data, target) in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        
        avg_loss = total_loss / (batch + 1)
        
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'perplexity': f'{math.exp(avg_loss):.2f}'
        })
        
        if batch % 100 == 0:
            wandb.log({
                "epoch": epoch,
                "batch": batch,
                "train_loss": avg_loss,
                "train_perplexity": math.exp(avg_loss),
                "learning_rate": optimizer.param_groups[0]['lr']
            })
    
    return avg_loss

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output.view(-1, output.size(-1)), target.view(-1)).item() * data.size(0)
    return total_loss / len(data_loader.dataset)

def generate_text(model, idx_to_word, device, start_text="The", max_length=50):
    model.eval()
    words = start_text.split()
    word_to_idx = {v: k for k, v in idx_to_word.items()}
    
    input_seq = [word_to_idx.get(w, word_to_idx['<unk>']) for w in words]
    input_seq = torch.tensor([input_seq]).to(device)
    
    generated_words = words.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_seq)
            next_word_idx = output[:, -1, :].argmax(dim=-1).item()
            next_word = idx_to_word[next_word_idx]
            generated_words.append(next_word)
            input_seq = torch.cat([input_seq, torch.tensor([[next_word_idx]]).to(device)], dim=1)
            
            if next_word == '<eos>':
                break
    
    return ' '.join(generated_words)

def train_mamba(batch_size=1024, seq_length=35, epochs=10, lr=0.001, d_model=256, d_state=16, d_conv=4, expand=2, num_layers=4):
    wandb.init(project="mamba-lm", config={
        "batch_size": batch_size,
        "seq_length": seq_length,
        "epochs": epochs,
        "learning_rate": lr,
        "d_model": d_model,
        "d_state": d_state,
        "d_conv": d_conv,
        "expand": expand,
        "num_layers": num_layers
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        print("Loading data...")
        train_data, val_data, test_data, vocab_size, idx_to_word = load_ptb_data(batch_size)
        train_loader, val_loader, test_loader = get_data_loaders(train_data, val_data, test_data, batch_size, seq_length)
        print(f"Data loaded. Vocabulary size: {vocab_size}")

        print("Initializing model...")
        model = MambaLM(vocab_size, d_model, d_state, d_conv, expand, num_layers).to(device)
        wandb.watch(model, log_freq=100)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            train_loss = train(model, train_loader, optimizer, criterion, device, epoch)
            val_loss = evaluate(model, val_loader, criterion, device)
            print(f'Epoch {epoch} - Train Loss: {train_loss:.4f}, '
                  f'Train Perplexity: {math.exp(train_loss):.2f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val Perplexity: {math.exp(val_loss):.2f}')
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_perplexity": math.exp(train_loss),
                "val_loss": val_loss,
                "val_perplexity": math.exp(val_loss),
                "learning_rate": scheduler.get_last_lr()[0]
            })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = 'mamba_model_best.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'idx_to_word': idx_to_word
                }, save_path)
                # torch.save(model.state_dict(), 'mamba_model_best.pt')
                wandb.save('mamba_model_best.pt')
            
            sample_text = generate_text(model, idx_to_word, device, start_text="The", max_length=50)
            wandb.log({
                "epoch": epoch,
                "generated_text": wandb.Html(sample_text.replace('\n', '<br>'))
            })
            
            scheduler.step()

        # model.load_state_dict(torch.load('mamba_model_best.pt'))
        checkpoint = torch.load('mamba_model_best.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        idx_to_word = checkpoint['idx_to_word']
        print("Model and idx_to_word loaded successfully.")
        ###
        test_loss = evaluate(model, test_loader, criterion, device)
        test_perplexity = math.exp(test_loss)
        print(f'Test Loss: {test_loss:.4f}, Test Perplexity: {test_perplexity:.2f}')
        wandb.log({"test_loss": test_loss, "test_perplexity": test_perplexity})

        return model, idx_to_word

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        wandb.finish()
        return None, None

    finally:
        wandb.finish()

# Run the training
if __name__ == "__main__":
    wandb.login()
    model, idx_to_word = train_mamba()