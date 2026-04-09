"""
Integrated Gradients attribution for analyzing jailbreak compliance flips.

Uses Captum to compute token-level attribution scores, measuring how each
input token contributes to the model's decision to comply vs refuse.

Note: Gradients through 4-bit quantized layers use straight-through estimators,
so attribution values are approximate but directionally correct.
"""

import torch
from captum.attr import LayerIntegratedGradients


def _make_forward_fn(model, comply_ids, refuse_ids):
    """Build a forward function: input_ids -> compliance score scalar.

    Used with LayerIntegratedGradients which hooks into the embedding layer
    to compute attributions. The function must accept the model's normal
    inputs (input_ids), not embeddings.
    """

    def forward_fn(input_ids):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :]  # last position
        comply_logit = logits[comply_ids].mean()
        refuse_logit = logits[refuse_ids].mean()
        return (comply_logit - refuse_logit).unsqueeze(0)

    return forward_fn


def get_compliance_token_ids(tokenizer):
    """Get token IDs for common compliance vs refusal first tokens."""
    comply_tokens = ["Sure", "Here", "Okay", "Below", "The", "To"]
    refuse_tokens = ["I", "Sorry", "As", "Unfortunately", "This"]

    comply_ids = []
    refuse_ids = []
    for t in comply_tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if ids:
            comply_ids.append(ids[0])
    for t in refuse_tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if ids:
            refuse_ids.append(ids[0])

    return torch.tensor(comply_ids), torch.tensor(refuse_ids)


def _find_user_span(tokenizer, raw_prompt, formatted_input_ids):
    """Find the [start, end) token range where raw_prompt appears in the chat-formatted IDs.

    Chat templates wrap the user content in boilerplate (BOS, system prompt,
    [INST]/[/INST], etc.). For visualization we want to highlight only the
    user-supplied content, not the template noise.

    Returns (start, end) tuple, or None if no match was found.
    """
    if raw_prompt is None:
        return None
    formatted_ids = formatted_input_ids[0].tolist()

    # Try a few tokenization variants — chat templates may insert a leading
    # space or newline that changes how the first user token gets encoded.
    variants = [
        raw_prompt,
        " " + raw_prompt,
        "\n" + raw_prompt,
    ]
    for variant in variants:
        raw_ids = tokenizer.encode(variant, add_special_tokens=False)
        n = len(raw_ids)
        if n == 0:
            continue
        for i in range(len(formatted_ids) - n + 1):
            if formatted_ids[i:i + n] == raw_ids:
                return (i, i + n)

    # Fuzzy fallback: match the longest suffix of raw_ids that appears in formatted_ids.
    # Useful if the very first token of the user content gets merged with a
    # preceding template token (common with BPE tokenizers).
    raw_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)
    for trim in range(1, min(4, len(raw_ids))):
        sub = raw_ids[trim:]
        n = len(sub)
        for i in range(len(formatted_ids) - n + 1):
            if formatted_ids[i:i + n] == sub:
                return (i, i + n)

    return None


def compute_attribution(model, tokenizer, prompt, comply_ids, refuse_ids,
                        n_steps=50, raw_prompt=None):
    """Compute Integrated Gradients attribution for input tokens.

    Uses LayerIntegratedGradients targeting the embedding layer. This computes
    the gradient of the compliance score w.r.t. the embedding layer's output,
    integrated along the interpolation path from baseline to actual embeddings.

    Args:
        model: The loaded language model.
        tokenizer: The model's tokenizer.
        prompt: Input prompt string (already formatted with chat template).
        comply_ids: Tensor of compliance token IDs.
        refuse_ids: Tensor of refusal token IDs.
        n_steps: Number of interpolation steps for IG (default 50).
        raw_prompt: Optional. The user-supplied content before chat templating;
            used to locate the user-content span for visualization purposes.

    Returns:
        dict with:
            - tokens: list of token strings
            - attributions: tensor of per-token attribution scores
            - compliance_score: scalar compliance score for this prompt
            - user_span: tuple (start, end) into tokens/attributions marking
              the user-supplied portion, or None if it could not be located
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    comply_ids = comply_ids.to(model.device)
    refuse_ids = refuse_ids.to(model.device)

    embedding_layer = model.get_input_embeddings()
    forward_fn = _make_forward_fn(model, comply_ids, refuse_ids)

    # LayerIntegratedGradients:
    # - forward_fn(input_ids) returns the compliance score
    # - It hooks into embedding_layer to get gradients of that layer's output
    # - baselines=None means zero embedding baseline (LIG handles this internally)
    lig = LayerIntegratedGradients(forward_fn, embedding_layer)
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=None,
        n_steps=n_steps,
        internal_batch_size=1,
    )

    # Sum over embedding dimension to get per-token score
    # attributions shape: (1, seq_len, hidden_dim)
    attr_scores = attributions.sum(dim=-1).squeeze(0)  # (seq_len,)

    # Normalize by L1 norm
    attr_scores = attr_scores / (attr_scores.abs().sum() + 1e-10)

    # Get token strings
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]

    # Locate the user-content span within the chat-formatted token sequence
    user_span = _find_user_span(tokenizer, raw_prompt, input_ids.cpu())

    # Compute compliance score
    with torch.no_grad():
        score = forward_fn(input_ids).item()

    return {
        "tokens": tokens,
        "attributions": attr_scores.detach().cpu(),
        "compliance_score": score,
        "user_span": user_span,
    }


def compare_attributions(attr_clean, attr_jailbreak):
    """Compare attribution maps between refused and jailbroken prompts.

    Since the prompts have different tokenizations, we report summary
    statistics rather than token-aligned diffs.

    Returns:
        dict with top contributing tokens for each prompt and score comparison.
    """
    def top_tokens(attr_dict, k=10):
        scores = attr_dict["attributions"]
        tokens = attr_dict["tokens"]
        # Restrict to user-supplied content if we know where it starts/ends.
        # Otherwise the top tokens are all Mistral system-prompt boilerplate.
        span = attr_dict.get("user_span")
        if span is not None:
            start, end = span
            scores = scores[start:end]
            tokens = tokens[start:end]
        top_idx = scores.abs().topk(min(k, len(scores))).indices
        return [(tokens[i], scores[i].item()) for i in top_idx]

    return {
        "clean_compliance_score": attr_clean["compliance_score"],
        "jailbreak_compliance_score": attr_jailbreak["compliance_score"],
        "score_shift": attr_jailbreak["compliance_score"] - attr_clean["compliance_score"],
        "clean_top_tokens": top_tokens(attr_clean),
        "jailbreak_top_tokens": top_tokens(attr_jailbreak),
    }
