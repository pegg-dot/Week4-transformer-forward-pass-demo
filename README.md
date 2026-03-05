# Transformer Forward Pass + Attention Explorer

An interactive educational demo that shows **how transformer attention works during a forward pass**.

This project includes:

- A **minimal transformer forward pass implementation**
- A **visual attention explorer** to inspect attention heads
- Tools for experimenting with **attention patterns and token interactions**

The goal is to make transformer mechanics **intuitive and visual**.

---

# Features

## Transformer Forward Pass

Implements the core pieces of a transformer block:

- token embeddings  
- positional embeddings  
- multi-head self-attention  
- attention weights  
- output projection  

---

## Attention Visualization

The web visualizer lets you:

- view **attention arcs between tokens**
- inspect different **attention heads**
- filter weak connections using a **threshold**
- show only **Top-K strongest arcs**
- auto-cycle through heads using **Play**

---

## Interactive Sentence Input

Users can input their own sentence to visualize attention behavior.

---

# What Each Attention Head Shows

This demo includes multiple heads that demonstrate common attention patterns.

| Head | Pattern | Description |
|-----|------|------|
| Head 0 | Local Context | Tokens attend strongly to nearby words |
| Head 1 | Self Attention | Tokens attend to themselves |
| Head 2 | Past Context | Tokens attend to earlier tokens |
| Head 3 | Boundary Anchor | Tokens attend to sentence start/end |

These illustrate **how different heads learn specialized roles**.

---

# Project Structure

```
Week4-transformer-forward-pass-demo
│
├── model.py
├── run_demo.py
├── bench.py
├── test_model.py
│
├── viz
│   ├── index.html
│   └── attn.json
│
└── README.md
```

---

# Setup

Clone the repository:

```bash
git clone https://github.com/pegg-dot/Week4-transformer-forward-pass-demo.git
cd Week4-transformer-forward-pass-demo
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install torch numpy
```

---

# Running the Demo

Run the forward pass:

```bash
python run_demo.py
```

This generates attention weights.

---

# Open the Attention Explorer

Open the visualization by opening this file in your browser:

```
viz/index.html
```

The interface lets you:

- inspect attention heads
- hover arcs to see attention weights
- filter weak connections
- animate head transitions

---

# Educational Purpose

This project is designed for:

- learning **transformer attention**
- visualizing **token relationships**
- experimenting with **multi-head attention**
- teaching **LLM architecture concepts**

---

# Future Improvements

Potential extensions:

- real GPT-2 attention extraction
- layer-by-layer visualization
- Q/K/V vector inspection
- attention heatmaps
- token embedding projections
- interactive transformer step debugger
