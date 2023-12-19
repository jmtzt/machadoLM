import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path


class Hparams:
    batch_size = 64
    block_size = 256
    max_iters = 10000
    eval_interval = 2000
    learning_rate = 3e-4
    eval_iters = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = "data/romance.txt"
    weights_path = "weights/romance.pt"
    num_embeddings = 384
    num_heads = 6
    num_layers = 6
    dropout = 0.2


# Set random seed for reproducibility
torch.manual_seed(1337)


def load_text(file_path):
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"No file found at {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def encode(text, stoi):
    return [stoi[c] for c in text]


def decode(tokens, itos):
    return "".join([itos[i] for i in tokens])


def get_batch(split, block_size, batch_size, data, device):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


class FeedForwardBlock(nn.Module):
    def __init__(self,
                 num_embeddings,
                 dropout,
                 expansion_factor=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, expansion_factor * num_embeddings),
            nn.ReLU(),
            nn.Linear(expansion_factor * num_embeddings, num_embeddings),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self,
                 head_size,
                 num_embeddings,
                 block_size,
                 dropout):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(num_embeddings, head_size, bias=False)
        self.query = nn.Linear(num_embeddings, head_size, bias=False)
        self.value = nn.Linear(num_embeddings, head_size, bias=False)
        # buffer is a tensor that is not a model parameter
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (batch,time,head_size)
        q = self.query(x)  # (batch,time,head_size)

        # (B, time, head_size) @ (B, head_size, time) -> (batch,time,time)
        # C is just used to scale the dot product, helps w/ softmax stability
        weights = q @ k.transpose(-1, -2) * C**(-0.5)
        # Lower triangular mask to prevent agg future tokens
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, time, time)
        weights = F.softmax(weights, dim=-1)  # (B, time, time)
        weights = self.dropout(weights)  # (B, time, time)

        # perform weighted aggregation of values
        v = self.value(x)  # (batch,time,head_size)
        return weights @ v  # (batch,time,head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 head_size,
                 num_embeddings,
                 block_size,
                 num_heads,
                 dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, num_embeddings, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(num_embeddings, num_embeddings)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.projection(out)


class TransformerBlock(nn.Module):
    def __init__(self,
                 num_embeddings,
                 block_size,
                 num_heads,
                 dropout):
        super().__init__()
        head_size = num_embeddings // num_heads
        self.attention_head = MultiHeadAttention(
            head_size, num_embeddings, block_size, num_heads, dropout)
        self.fc = FeedForwardBlock(num_embeddings, dropout=dropout)
        self.ln1 = nn.LayerNorm(num_embeddings)
        self.ln2 = nn.LayerNorm(num_embeddings)

    def forward(self, x):
        x = x + self.attention_head(self.ln1(x))
        x = x + self.fc(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 block_size,
                 num_embeddings,
                 num_heads,
                 num_layers,
                 dropout):
        super().__init__()
        # The embedding table is a square matrix of size vocab_size
        # Each row of the matrix is a vector representation of a token
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(
            vocab_size, num_embeddings)

        # The position embedding table is a rectangular matrix of size
        # block_size x num_embeddings
        self.position_embedding_table = nn.Embedding(
            block_size, num_embeddings)

        # Define the attention head
        # 4 heads of size num_embeddings//4 to get num_embeddings via concat
        self.blocks = nn.Sequential(
            *[TransformerBlock(num_embeddings,
                               block_size,
                               num_heads=num_heads,
                               dropout=dropout) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(num_embeddings)

        self.fc = FeedForwardBlock(num_embeddings, dropout=dropout)

        self.lm_head = nn.Linear(num_embeddings, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (batch,time) tensor of integers
        token_embed = self.token_embedding_table(idx)  # (batch,time,n_embed)
        pos_embed = self.position_embedding_table(
            torch.arange(T, device=idx.device))  # (time,n_embed)
        # Why do we need pos embed? There's no notion of space in the model
        # Thus we need to inject this information somehow
        x = token_embed + pos_embed  # (batch,time,n_embed)
        # After encoding the tokens and positions, we can now apply attention
        x = self.blocks(x)  # (batch,time,n_embed)
        x = self.ln_final(x)  # (batch,time,n_embed) -> LayerNorm
        # After attention, we can apply a feedforward block to each token
        # Once data is aggregated, we apply a FC layer to process it further
        x = self.fc(x)  # (batch,time,n_embed)
        logits = self.lm_head(x)  # (batch,time,Vocab_size)

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
            # crop index to the last block_size tokens
            # otherwise it might be too long for our pos embedding table
            idx_crop = idx[:, -self.block_size:]
            logits, _ = self(idx_crop)  # get preds
            logits = logits[:, -1, :]  # focus only on the last token
            probs = F.softmax(logits, dim=-1)  # get probs
            idx_next = torch.multinomial(probs, num_samples=1)  # sample
            idx = torch.cat([idx, idx_next], dim=1)  # add to sequence
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
                f"step {iter}:",
                f"train loss {losses['train']: .4f}",
                f"val loss {losses['val']: .4f}")
        xb, yb = get_batch('train', block_size, batch_size, train_data, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def save_model(model, file_path):
    if not Path(file_path).parent.is_dir():
        Path(file_path).parent.mkdir(parents=True)
    torch.save(model.state_dict(), file_path)


def load_model(model, file_path, device):
    model.load_state_dict(torch.load(file_path, map_location=device))
    return model


if __name__ == "__main__":
    text = load_text(Hparams.data_path)
    vocab = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for i, c in enumerate(vocab)}
    tokens = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = int(0.9 * len(tokens))
    train_data, val_data = tokens[:n], tokens[n:]

    print(f'Length of vocab: {len(vocab)}')
    model = LanguageModel(vocab_size=len(vocab),
                          block_size=Hparams.block_size,
                          num_embeddings=Hparams.num_embeddings,
                          num_heads=Hparams.num_heads,
                          num_layers=Hparams.num_layers,
                          dropout=Hparams.dropout).to(Hparams.device)

    # Pre-training generation
    context = torch.zeros((1, 1), dtype=torch.long, device=Hparams.device)
    generated_output = decode(model.generate(context, 100)[0].tolist(), itos)
    print(f'Generated text before training: {generated_output}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=Hparams.learning_rate)
    model = train_model(model, optimizer, Hparams.max_iters,
                        Hparams.eval_interval, Hparams.eval_iters,
                        Hparams.block_size, Hparams.batch_size,
                        train_data, val_data, Hparams.device)

    # Post-training generation
    context = torch.zeros((1, 1), dtype=torch.long, device=Hparams.device)
    generated_output = decode(model.generate(context, 100)[0].tolist(), itos)
    print(f'Generated text after training: {generated_output}')

    save_model(model, Hparams.weights_path)
