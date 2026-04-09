"""
Evaluation metrics for validating XAI attribution results.

- Ablation test: mask top-K attributed tokens and check if compliance flips
- Kendall's Tau: rank correlation between different attribution methods
"""

import torch
from scipy.stats import kendalltau


def ablation_test(model, tokenizer, prompt, attribution_scores, token_ids,
                  comply_ids, refuse_ids, k_values=None, mask_token_id=None):
    """Mask top-K attributed tokens and measure compliance score change.

    If the attribution is correct, masking the most important tokens should
    flip the model back toward refusal (for jailbreak prompts) or toward
    compliance (for clean prompts).

    Args:
        model: The language model.
        tokenizer: The model's tokenizer.
        prompt: The original prompt string.
        attribution_scores: Tensor of per-token attribution scores.
        token_ids: Tensor of token IDs for the prompt.
        comply_ids: Compliance token IDs.
        refuse_ids: Refusal token IDs.
        k_values: List of K values to test (default [1, 3, 5, 10, 20]).
        mask_token_id: Token ID to use for masking (default: tokenizer.unk_token_id).

    Returns:
        dict mapping K -> compliance score after masking top-K tokens.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    if mask_token_id is None:
        mask_token_id = tokenizer.unk_token_id or 0

    comply_ids = comply_ids.to(model.device)
    refuse_ids = refuse_ids.to(model.device)

    def compliance_score(ids):
        ids = ids.to(model.device)
        with torch.no_grad():
            out = model(ids)
            logits = out.logits[0, -1, :]
        return logits[comply_ids].mean().item() - logits[refuse_ids].mean().item()

    # Baseline score (no masking)
    original_ids = token_ids.unsqueeze(0) if token_ids.dim() == 1 else token_ids
    baseline = compliance_score(original_ids)

    # Rank tokens by absolute attribution (highest first)
    ranked = attribution_scores.abs().argsort(descending=True)

    results = {"baseline": baseline, "scores": {}}
    for k in k_values:
        k = min(k, len(ranked))
        masked_ids = original_ids.clone()
        top_k_positions = ranked[:k]
        masked_ids[0, top_k_positions] = mask_token_id
        score = compliance_score(masked_ids)
        results["scores"][k] = score

    return results


def kendall_tau(attributions_a, attributions_b):
    """Compute Kendall's Tau rank correlation between two attribution vectors.

    Both vectors should be the same length (same tokenization).

    Returns:
        (tau, p_value)
    """
    a = attributions_a.numpy() if torch.is_tensor(attributions_a) else attributions_a
    b = attributions_b.numpy() if torch.is_tensor(attributions_b) else attributions_b
    tau, p = kendalltau(a, b)
    return tau, p
