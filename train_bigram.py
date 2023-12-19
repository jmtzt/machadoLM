import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path


class Config:
    batch_size = 32
    block_size = 8
    max_iters = 3000
    eval_interval = 3001
    learning_rate = 1e-2
    eval_iters = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = "data/romance.txt"


# Set random seed for reproducibility
torch.manual_seed(1337)


def load_text(file_path):
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"No file found at {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


text = load_text(Config.data_path)


def encode(text, stoi):
    return [stoi[c] for c in text]


def decode(tokens, itos):
    return "".join([itos[i] for i in tokens])


# Prepare data
vocab = sorted(list(set(text)))
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for i, c in enumerate(vocab)}
tokens = torch.tensor(encode(text, stoi), dtype=torch.long)
n = int(0.9 * len(tokens))
train_data, val_data = tokens[:n], tokens[n:]


def get_batch(split, block_size, batch_size, data, device):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # The embedding table is a square matrix of size vocab_size
        # Each row of the matrix is a vector representation of a token
        self.token_embedding_table = nn.Embedding(
            vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def estimate_loss(model, eval_iters, block_size,
                  batch_size, train_data, val_data, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        data = train_data if split == 'train' else val_data
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, block_size, batch_size, data, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(model, optimizer, max_iters, eval_interval,
                eval_iters, block_size, batch_size, train_data,
                val_data, device):
    for iter in tqdm(range(max_iters)):
        if iter % eval_interval == 0:
            losses = estimate_loss(
                model, eval_iters, block_size, batch_size,
                train_data, val_data, device)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch('train', block_size, batch_size, train_data, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


# Main execution
if __name__ == "__main__":
    model = BigramLanguageModel(len(vocab)).to(Config.device)

    context = torch.zeros((1, 1), dtype=torch.long, device=Config.device)
    generated_output = decode(model.generate(context, 100)[0].tolist(), itos)
    print(f'Generated text before training: {generated_output}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    model = train_model(model, optimizer, Config.max_iters,
                        Config.eval_interval, Config.eval_iters,
                        Config.block_size, Config.batch_size,
                        train_data, val_data, Config.device)

    # Post-training generation
    context = torch.zeros((1, 1), dtype=torch.long, device=Config.device)
    generated_output = decode(model.generate(context, 100)[0].tolist(), itos)
    print(f'Generated text after training: {generated_output}')
