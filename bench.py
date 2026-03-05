import time
import torch
from model import TransformerDemo

torch.manual_seed(0)

device = "cpu"

model = TransformerDemo(
    vocab_size=200,
    d_model=64,
    n_heads=4,
    d_ff=128,
    max_len=128
).to(device)

tokens = torch.randint(0, 200, (8, 32), device=device)

# Warmup runs (helps stabilize timing)
for _ in range(20):
    model(tokens)

N = 300

start = time.perf_counter()

for _ in range(N):
    model(tokens)

end = time.perf_counter()

avg_ms = (end - start) * 1000 / N

print(f"Average forward pass: {avg_ms:.3f} ms (batch=8, seq=32)")