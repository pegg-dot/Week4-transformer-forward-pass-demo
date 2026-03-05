import torch
from model import TransformerDemo

torch.manual_seed(0)

model = TransformerDemo(
    vocab_size=50,
    d_model=32,
    n_heads=4,
    d_ff=64,
    max_len=64
)

# batch=2, sequence length=8
tokens = torch.randint(0, 50, (2, 8))

logits, dbg = model(tokens, debug=True)

print("logits shape:", logits.shape)
print("attention shape:", dbg["attn"].shape)
