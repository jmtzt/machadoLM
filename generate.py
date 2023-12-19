import torch
from train import Hparams, LanguageModel, load_model, decode, load_text

if __name__ == '__main__':
    text = load_text(Hparams.data_path)
    vocab = sorted(list(set(text)))

    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for i, c in enumerate(vocab)}

    model = LanguageModel(vocab_size=len(vocab),
                          block_size=Hparams.block_size,
                          num_embeddings=Hparams.num_embeddings,
                          num_heads=Hparams.num_heads,
                          num_layers=Hparams.num_layers,
                          dropout=Hparams.dropout).to(Hparams.device)

    model = load_model(model, Hparams.weights_path, device=Hparams.device)
    context = torch.zeros((1, 1), dtype=torch.long, device=Hparams.device)
    generated_output = decode(model.generate(context, 1000)[0].tolist(), itos)
    print(f'Generated text after training: {generated_output}')
