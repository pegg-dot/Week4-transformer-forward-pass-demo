import json
import torch
import sys
import os

# allow import from parent folder
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model import TransformerDemo

torch.manual_seed(0)

# Create model
model = TransformerDemo(
    vocab_size=50,
    d_model=32,
    n_heads=4,
    d_ff=64,
    max_len=64
)

# Fake tokens
tokens = torch.randint(0, 50, (1, 8))

# Run model
_, dbg = model(tokens, debug=True)

attn = dbg["attn"][0]  # remove batch dimension -> [H, T, T]

# convert to Python lists
attn_list = attn.tolist()

data = {
    "tokens": tokens[0].tolist(),
    "attention": attn_list
}

# save JSON
out_path = os.path.join(os.path.dirname(__file__), "attn.json")

with open(out_path, "w") as f:
    json.dump(data, f, indent=2)

print("Exported attention to:", out_path)