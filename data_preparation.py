import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer

def load_and_preprocess_data(max_length=128, stride=64):
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
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
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch")
    
    return tokenized_dataset, tokenizer

def create_dataloaders(dataset, batch_size=32):
    train_dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset["validation"], batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(dataset["test"], batch_size=batch_size)
    
    return train_dataloader, val_dataloader, test_dataloader

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
        break