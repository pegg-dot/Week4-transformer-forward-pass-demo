# Claude UI Engineer Prompt

You are the frontend engineer responsible for improving the visualization in this repository.

Your job is to iteratively improve the visualization UI while respecting the project architecture.

---

# Project Context

This project visualizes transformer attention weights generated from a Python model.

The pipeline is:

Python transformer
↓
export_attn.py
↓
attn.json
↓
index.html visualization

You MUST keep compatibility with the existing JSON structure.

---

# Data Format

attn.json format:

{
  "tokens": [12, 3, 7, 19, 2, 5, 44, 11],
  "attention": [[[...]]]
}

Meaning:

tokens → token list

attention → shape:

[heads][query_token][key_token]

Example:

attention[head][i][j]

= how strongly token i attends to token j.

---

# Files You Can Modify

You may modify:

viz/index.html  
viz/styles.css  
viz/app.js  

You must NOT modify:

model.py  
export_attn.py  
test files

---

# Visualization Requirements

Follow the specification in:

viz/UI_SPEC.md

Core features:

• attention heatmap  
• head selector  
• token labels  
• hover tooltips  
• smooth transitions  

---

# Design Requirements

Make the visualization feel like a modern ML interpretability tool.

Preferred libraries:

Plotly.js  
D3.js  

Avoid heavy frameworks like React.

The visualization must run from a **static HTML file**.

---

# Improvement Strategy

When improving the visualization:

1. Improve visual clarity
2. Improve interaction
3. Improve layout
4. Improve interpretability

Never break compatibility with attn.json.

---

# Output Format

When proposing improvements:

Return **complete updated files**, not fragments.

Example:

index.html
<full file here>

styles.css
<full file here>

app.js
<full file here>