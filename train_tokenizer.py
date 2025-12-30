from tokenizers import ByteLevelBPETokenizer
import os

os.makedirs("tokenizer", exist_ok=True)

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["data/train.txt"],
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

tokenizer.save_model("tokenizer")
print("Tokenizer trained.")
