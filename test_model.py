import torch
from model import TransformerDemo


def test_shapes():
    torch.manual_seed(0)

    model = TransformerDemo(
        vocab_size=50,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_len=64
    )

    tokens = torch.randint(0, 50, (2, 8))

    logits, dbg = model(tokens, debug=True)

    assert logits.shape == (2, 8, 50)
    assert "attn" in dbg


def test_no_nans():
    torch.manual_seed(0)

    model = TransformerDemo(
        vocab_size=50,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_len=64
    )

    tokens = torch.randint(0, 50, (2, 8))

    logits, dbg = model(tokens, debug=True)

    assert not torch.isnan(logits).any()
    assert not torch.isnan(dbg["attn"]).any()


def test_attention_rows_sum_to_one():
    torch.manual_seed(0)

    model = TransformerDemo(
        vocab_size=50,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_len=64
    )

    tokens = torch.randint(0, 50, (2, 8))

    _, dbg = model(tokens, debug=True)

    attn = dbg["attn"]  # [B, H, T, T]

    row_sums = attn.sum(dim=-1)

    ones = torch.ones_like(row_sums)

    assert torch.allclose(row_sums, ones, atol=1e-5)


def test_deterministic_with_seed():
    torch.manual_seed(123)

    model1 = TransformerDemo(
        vocab_size=50,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_len=64
    )

    tokens = torch.randint(0, 50, (2, 8))

    out1, _ = model1(tokens)

    torch.manual_seed(123)

    model2 = TransformerDemo(
        vocab_size=50,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_len=64
    )

    out2, _ = model2(tokens)

    assert torch.allclose(out1, out2, atol=1e-6)