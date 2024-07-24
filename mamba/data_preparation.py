import torch
from datasets import load_dataset, concatenate_datasets
from transformers import GPT2TokenizerFast
from typing import Tuple, Dict, Any
import random

def load_and_preprocess_data(max_length: int = 128, stride: int = 64, dataset_name: str = "wikitext", dataset_config: str = "wikitext-2-v1") -> Tuple[Dict[str, Any], GPT2TokenizerFast]:
    # Load the primary dataset
    dataset = load_dataset(dataset_name, dataset_config)
    
    # Optionally, load and combine additional datasets
    # Uncomment and modify these lines to include more datasets
    # bookcorpus = load_dataset("bookcorpus", split="train")
    # openwebtext = load_dataset("openwebtext", split="train")
    # dataset['train'] = concatenate_datasets([dataset['train'], bookcorpus, openwebtext])
    # dataset['train'] = concatenate_datasets([dataset['train'], bookcorpus])
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
            return_length=True,
            stride=stride,
        )
        
        input_batch = []
        for length, input_ids in zip(tokenized_inputs["length"], tokenized_inputs["input_ids"]):
            if length == max_length:
                input_batch.append(input_ids)
        
        return {"input_ids": input_batch}
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_dataset.set_format(type="torch")
    
    return tokenized_dataset, tokenizer

def create_dataloaders(dataset: Dict[str, Any], batch_size: int = 32) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset["validation"], batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(dataset["test"], batch_size=batch_size)
    
    return train_dataloader, val_dataloader, test_dataloader

def random_masking(input_ids: torch.Tensor, mask_prob: float = 0.15) -> torch.Tensor:
    """Randomly mask input tokens for MLM-style training."""
    mask = torch.rand(input_ids.shape) < mask_prob
    input_ids[mask] = tokenizer.mask_token_id
    return input_ids

def token_shuffling(input_ids: torch.Tensor, shuffle_prob: float = 0.1) -> torch.Tensor:
    """Randomly shuffle a small portion of input tokens."""
    if random.random() < shuffle_prob:
        length = input_ids.size(0)
        shuffle_idx = torch.randperm(length)
        input_ids = input_ids[shuffle_idx]
    return input_ids

if __name__ == "__main__":
    dataset, tokenizer = load_and_preprocess_data()
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(dataset)
    
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    for batch in train_dataloader:
        print("Sample batch shape:", batch["input_ids"].shape)
        print("Sample input:")
        print(tokenizer.decode(batch["input_ids"][0]))
        
        # Demonstrate data augmentation techniques
        masked_input = random_masking(batch["input_ids"][0].clone())
        print("\nMasked input:")
        print(tokenizer.decode(masked_input))
        
        shuffled_input = token_shuffling(batch["input_ids"][0].clone())
        print("\nShuffled input:")
        print(tokenizer.decode(shuffled_input))
        
        break