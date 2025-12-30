"""
Train a small language model from scratch using HuggingFace
This script trains a GPT-2 style model on any text dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os

# Configuration
class Config:
    # Model architecture
    vocab_size = 50257  # GPT-2 tokenizer vocab size
    n_positions = 512   # Max sequence length
    n_embd = 512        # Embedding dimension
    n_layer = 6         # Number of transformer layers
    n_head = 8          # Number of attention heads
    
    # Training parameters
    batch_size = 8
    num_epochs = 3
    learning_rate = 5e-4
    warmup_steps = 500
    max_steps = 10000
    gradient_accumulation_steps = 4
    
    # Data
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    block_size = 512
    
    # Checkpointing
    save_dir = "./llm_checkpoints"
    save_steps = 1000

def prepare_dataset(tokenizer, config):
    """Load and tokenize the dataset"""
    print(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, config.dataset_config)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )
    
    def group_texts(examples):
        """Concatenate and chunk texts into blocks"""
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        
        # Drop the last chunk if it's smaller than block_size
        total_length = (total_length // config.block_size) * config.block_size
        
        result = {
            k: [t[i:i + config.block_size] 
                for i in range(0, total_length, config.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    print("Grouping texts into blocks...")
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        desc="Grouping texts"
    )
    
    return lm_dataset

def train(model, train_dataloader, optimizer, scheduler, config, device):
    """Training loop"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(range(config.max_steps), desc="Training")
    
    step = 0
    for epoch in range(config.num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            total_loss += loss.item()
            
            # Update weights
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": total_loss / step,
                    "lr": scheduler.get_last_lr()[0]
                })
                
                # Save checkpoint
                if step % config.save_steps == 0:
                    save_checkpoint(model, optimizer, step, config)
                
                if step >= config.max_steps:
                    break
        
        if step >= config.max_steps:
            break
    
    return model

def save_checkpoint(model, optimizer, step, config):
    """Save model checkpoint"""
    os.makedirs(config.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.save_dir, f"checkpoint-{step}")
    print(f"\nSaving checkpoint to {checkpoint_path}")
    model.save_pretrained(checkpoint_path)

def main():
    # Setup
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    lm_dataset = prepare_dataset(tokenizer, config)
    
    # Create dataloader
    train_dataloader = DataLoader(
        lm_dataset["train"],
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda x: {
            "input_ids": torch.stack([torch.tensor(d["input_ids"]) for d in x]),
            "attention_mask": torch.stack([torch.tensor(d["attention_mask"]) for d in x]),
            "labels": torch.stack([torch.tensor(d["labels"]) for d in x])
        }
    )
    
    # Initialize model
    print("Initializing model...")
    model_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
    )
    model = GPT2LMHeadModel(model_config)
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )
    
    # Train
    print("\nStarting training...")
    model = train(model, train_dataloader, optimizer, scheduler, config, device)
    
    # Save final model
    final_path = os.path.join(config.save_dir, "final_model")
    print(f"\nSaving final model to {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print("\nTraining complete!")
    print(f"Model saved to {final_path}")
    
    # Generate sample text
    print("\nGenerating sample text...")
    model.eval()
    prompt = "The future of artificial intelligence"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()
