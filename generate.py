import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from model import MiniGPT
import sys

# 1. Setup Device and Model
# Added checks for CUDA as well, just in case
if torch.backends.mps.is_available():
    DEVICE = "mps" 
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Loading model on {DEVICE}...")

tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

# Initialize and load model
model = MiniGPT(vocab_size=tokenizer.get_vocab_size()).to(DEVICE)
model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
model.eval()

# 2. Configuration for generation
MAX_NEW_TOKENS = 100   # How much the bot can talk per turn
TEMPERATURE = 0.8      # Higher = more creative, Lower = more deterministic
TOP_K = 50             # Keep only top 50 likely next tokens
BLOCK_SIZE = 256       # The max context length your model was trained with (adjust this!)

def generate_response(prompt_tokens):
    """
    Generates text using streaming and sampling.
    """
    curr_ids = torch.tensor([prompt_tokens], device=DEVICE)
    
    # We store the generated tokens here to return them later to history
    generated_ids = []

    print("\nBot: ", end='', flush=True)

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            # Crop context to ensure we don't exceed model's block size
            # If your model crashes with index out of bounds, check your BLOCK_SIZE
            cond_ids = curr_ids[:, -BLOCK_SIZE:]

            # Get predictions
            logits = model(cond_ids)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] 

            # Apply Temperature (controls randomness)
            logits = logits / TEMPERATURE

            # Apply Top-K filtering (optional, prevents garbage text)
            if TOP_K > 0:
                v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution (instead of argmax)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to current sequence
            curr_ids = torch.cat((curr_ids, next_token), dim=1)
            generated_ids.append(next_token.item())

            # Decode and print just this token (Streaming effect)
            # decode([token]) might look broken for partial utf-8 bytes, 
            # but usually works fine for BPE.
            token_str = tokenizer.decode([next_token.item()])
            print(token_str, end='', flush=True)

            # Optional: Stop if the model generates a specific 'End of Text' token
            # if next_token.item() == tokenizer.token_to_id("<|endoftext|>"):
            #     break

    print("\n") # New line after generation finishes
    return generated_ids

# 3. The Chat Loop
print("------------------------------------------------")
print("Chatbot Ready! Type 'quit' or 'exit' to stop.")
print("------------------------------------------------")

# We keep a running history of tokens to maintain context
conversation_history = []

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    # Encode user input
    # Adding a delimiter like "\n" helps the model understand turn-taking
    user_ids = tokenizer.encode("\nUser: " + user_input + "\nBot:").ids
    
    # Add to history
    conversation_history.extend(user_ids)

    # Generate response
    # We feed the whole history so the model "remembers" previous turns
    # (The generate function handles cropping if it gets too long)
    bot_tokens = generate_response(conversation_history)
    
    # Add bot response to history
    conversation_history.extend(bot_tokens)