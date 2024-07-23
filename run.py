import torch
import argparse
from transformers import GPT2Tokenizer
from mamba_model import create_mamba_model

class MambaInference:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path): ## Make sure that this config matches your model
        vocab_size = len(self.tokenizer)
        d_model = 512  # Changed from 256
        n_layer = 8    # Changed from 4
        d_state = 64   # Changed from 16
        d_conv = 4
        expand = 2

        model = create_mamba_model(vocab_size, d_model, n_layer, d_state, d_conv, expand)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def generate_next_word(self, input_text, temperature=1.0, top_k=50, top_p=0.95):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(-1, top_k_indices, top_k_logits)
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(-1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
    
        return self.tokenizer.decode(next_token[0])

    def generate_sequence(self, input_text, max_length=50, temperature=1.0, top_k=50, top_p=0.95):
        current_text = input_text
        for _ in range(max_length):
            next_word = self.generate_next_word(current_text, temperature, top_k, top_p)
            if next_word == self.tokenizer.eos_token:
                break
            current_text += ' ' + next_word
        return current_text

def main():
    parser = argparse.ArgumentParser(description="Mamba model inference")
    parser.add_argument("--model_path", type=str, default = "mamba_lm.pth", help="Path to the trained model")
    parser.add_argument("--input_text", type=str, required=True, help="Input text for generation")
    parser.add_argument("--mode", type=str, choices=['next_word', 'sequence'], required=True, help="Generation mode")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of generated sequence")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering parameter")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p filtering parameter")
    args = parser.parse_args()

    inference = MambaInference(args.model_path)
    
    if args.mode == 'next_word':
        next_word = inference.generate_next_word(
            args.input_text, 
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print(f"Input: {args.input_text}")
        print(f"Next word: {next_word}")
    else:  # sequence mode
        generated_text = inference.generate_sequence(
            args.input_text, 
            max_length=args.max_length, 
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print(f"Input: {args.input_text}")
        print(f"Generated sequence: {generated_text}")

if __name__ == "__main__":
    main()
    # before usage, make sure that the model config in load_model() matches your trained model configuration. otherwise, there will be a dimension mismatch.
    # sample usage:
    # for next word prediction: python run.py --input_text "Your input text here" --mode next_word --temperature 0.7 --top_k 50 --top_p 0.95
    # for sequence generation: python run.py --input_text "Your input text here" --mode sequence --max_length 100 --temperature 0.7 --top_k 50 --top_p 0.95