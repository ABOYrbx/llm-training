import torch
from torch.utils.data import DataLoader
from tokenizers import ByteLevelBPETokenizer
from model import MiniGPT
from dataset import TextDataset
from tqdm import tqdm
from datasets import load_dataset

SEQ_LEN = 256
BATCH_SIZE = 16
EPOCHS = 10
LR = 3e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("trentmkelly/autotrain-data-roblox-usernames")
text = "\n".join(ds["train"]["autotrain_text"])

tokens = tokenizer.encode(text).ids
dataset = TextDataset(tokens, SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = MiniGPT(vocab_size=tokenizer.get_vocab_size()).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(EPOCHS):
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        batch = batch.to(DEVICE)

        logits = model(batch[:, :-1])
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "model.pt")

print("Training complete.")
