"""
Activation tracing and patching for jailbreak mechanistic interpretability.

Uses nnsight to:
1. Cache hidden states at every layer for clean vs jailbreak prompts
2. Compute divergence heatmaps (cosine distance per layer × position)
3. Perform activation patching to causally identify compliance-critical layers
4. Logit lens: project each layer's hidden state to vocabulary space

Works with Unsloth-loaded Mistral models via nnsight's NNsight wrapper.

IMPORTANT: Inside nnsight trace contexts, all module access must go through
the wrapped model proxy (e.g., wrapped.model.layers[i]), NOT the raw model.
"""

import logging

import torch
import torch.nn.functional as F
from nnsight import NNsight

logger = logging.getLogger(__name__)


# Candidate attribute paths to walk when locating components in the model
# tree. Covers plain HF CausalLM models, Unsloth wrappers, and the multimodal
# Mistral3ForConditionalGeneration wrapper used by Mistral Small 3.1.
_CANDIDATE_PATHS = [
    (),
    ("model",),
    ("language_model",),
    ("language_model", "model"),
    ("model", "language_model"),
    ("model", "language_model", "model"),
]


def _traverse(obj, path):
    """Walk a tuple of attribute names from obj."""
    for a in path:
        obj = getattr(obj, a)
    return obj


def _find_attr_parent_path(model, attr_name):
    """Return the tuple-of-attrs path whose endpoint has `attr_name`, or None."""
    for path in _CANDIDATE_PATHS:
        obj = model
        ok = True
        for a in path:
            if not hasattr(obj, a):
                ok = False
                break
            obj = getattr(obj, a)
        if ok and hasattr(obj, attr_name):
            return path
    return None


def _get_model_info(model):
    """Detect model structure and return component paths.

    Returns a dict with:
        - n_layers: int
        - layers_path: tuple of attr names; the layers ModuleList lives at
          ``model.<layers_path>.layers``
        - norm_path: tuple or None; final norm at ``model.<norm_path>.norm``
        - lm_head_path: tuple or None; lm_head at ``model.<lm_head_path>.lm_head``
    """
    layers_path = _find_attr_parent_path(model, "layers")
    if layers_path is None:
        attrs = [a for a in dir(model) if not a.startswith("_")][:40]
        raise AttributeError(
            f"Cannot find transformer layers in model. "
            f"Class: {type(model).__name__}, top-level attrs: {attrs}"
        )

    layers_parent = _traverse(model, layers_path)
    n_layers = len(layers_parent.layers)

    norm_path = _find_attr_parent_path(model, "norm")
    lm_head_path = _find_attr_parent_path(model, "lm_head")

    return {
        "n_layers": n_layers,
        "layers_path": layers_path,
        "norm_path": norm_path,
        "lm_head_path": lm_head_path,
    }


def _get_wrapped_layer(wrapped, i, info):
    """Get the i-th layer proxy from the wrapped model."""
    return _traverse(wrapped, info["layers_path"]).layers[i]


def _get_wrapped_lm_head(wrapped, info):
    """Get the lm_head proxy from the wrapped model (or None)."""
    if info["lm_head_path"] is None:
        return None
    return _traverse(wrapped, info["lm_head_path"]).lm_head


def _get_wrapped_norm(wrapped, info):
    """Get the final norm proxy from the wrapped model (or None)."""
    if info["norm_path"] is None:
        return None
    return _traverse(wrapped, info["norm_path"]).norm


def _get_raw_lm_head(model, info):
    """Get the raw (non-proxy) lm_head module from the model (or None)."""
    if info["lm_head_path"] is None:
        return None
    return _traverse(model, info["lm_head_path"]).lm_head


def _get_raw_norm(model, info):
    """Get the raw (non-proxy) final norm module from the model (or None)."""
    if info["norm_path"] is None:
        return None
    return _traverse(model, info["norm_path"]).norm


def _last_pos(t):
    """Get the last token position's hidden state from a saved layer output.

    nnsight's ``layer.output[0].save()`` returns a 2D ``(seq_len, hidden_dim)``
    tensor for this model (the batch dim is already squeezed). We also handle
    the 3D ``(batch, seq_len, hidden_dim)`` case defensively in case nnsight's
    behavior shifts across versions.
    """
    if t.dim() == 3:
        return t[0, -1, :]
    if t.dim() == 2:
        return t[-1, :]
    raise ValueError(f"Unexpected hidden-state tensor shape: {tuple(t.shape)}")




def cache_activations(model, tokenizer, prompt):
    """Run the model and cache hidden states at every layer's output.

    Args:
        model: The language model.
        tokenizer: The model's tokenizer.
        prompt: Input prompt string.

    Returns:
        dict with:
            - hidden_states: list of tensors, one per layer, shape (seq_len, hidden_dim)
            - input_ids: the tokenized input
            - tokens: list of token strings
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    info = _get_model_info(model)
    n_layers = info["n_layers"]

    wrapped = NNsight(model)
    saved = []

    with torch.no_grad():
        with wrapped.trace(input_ids):
            for i in range(n_layers):
                layer = _get_wrapped_layer(wrapped, i, info)
                h = layer.output[0].save()
                saved.append(h)

    hidden_states = [s.squeeze(0).cpu() for s in saved]  # each: (seq_len, hidden_dim)
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]

    return {
        "hidden_states": hidden_states,
        "input_ids": input_ids.cpu(),
        "tokens": tokens,
    }


def compute_divergence_heatmap(cache_clean, cache_jailbreak):
    """Compute cosine distance between clean and jailbreak hidden states.

    Since the two prompts may have different sequence lengths, we compute:
    1. Per-layer divergence at the LAST token position (the prediction position)
    2. Per-layer average divergence across all positions of the shorter sequence

    Args:
        cache_clean: Output from cache_activations for the clean (refused) prompt.
        cache_jailbreak: Output from cache_activations for the jailbreak prompt.

    Returns:
        dict with:
            - last_pos_divergence: tensor of shape (n_layers,) — cosine distance
              at last token position per layer
            - layer_avg_divergence: tensor of shape (n_layers,) — average cosine
              distance across aligned positions
            - n_layers: number of layers
    """
    hs_clean = cache_clean["hidden_states"]
    hs_jailbreak = cache_jailbreak["hidden_states"]
    n_layers = len(hs_clean)

    last_pos_div = []
    layer_avg_div = []

    for i in range(n_layers):
        hc = hs_clean[i]    # (seq_len_clean, hidden_dim)
        hj = hs_jailbreak[i]  # (seq_len_jb, hidden_dim)

        # Last position divergence
        cos_sim = F.cosine_similarity(hc[-1:], hj[-1:], dim=-1)
        last_pos_div.append(1.0 - cos_sim.item())

        # Average over aligned positions (min length)
        min_len = min(hc.shape[0], hj.shape[0])
        cos_sims = F.cosine_similarity(hc[:min_len], hj[:min_len], dim=-1)
        layer_avg_div.append((1.0 - cos_sims).mean().item())

    return {
        "last_pos_divergence": torch.tensor(last_pos_div),
        "layer_avg_divergence": torch.tensor(layer_avg_div),
        "n_layers": n_layers,
    }


def activation_patch_by_layer(model, tokenizer, clean_prompt, jailbreak_prompt,
                              comply_ids, refuse_ids):
    """Patch activations layer-by-layer to find compliance-critical layers.

    For each layer: run the clean prompt but replace that layer's output
    (at the last token position) with the jailbreak prompt's hidden state.
    Measure how much the output shifts toward compliance.

    Args:
        model: The language model.
        tokenizer: The model's tokenizer.
        clean_prompt: The original prompt that gets refused.
        jailbreak_prompt: The mutated prompt that gets compliance.
        comply_ids: Tensor of compliance token IDs.
        refuse_ids: Tensor of refusal token IDs.

    Returns:
        dict with:
            - baseline_score: compliance score with no patching (clean prompt)
            - jailbreak_score: compliance score of the jailbreak prompt
            - patched_scores: tensor of shape (n_layers,) — compliance score
              when patching each layer
            - causal_effect: tensor of shape (n_layers,) — how much each
              layer's patch shifts the score toward jailbreak
            - n_layers: number of layers
    """
    info = _get_model_info(model)
    n_layers = info["n_layers"]

    comply_ids = comply_ids.to(model.device)
    refuse_ids = refuse_ids.to(model.device)

    def compliance_score(logits):
        """Compute comply - refuse logit gap."""
        return logits[comply_ids].mean().item() - logits[refuse_ids].mean().item()

    # First, cache the jailbreak hidden states via nnsight
    logger.info("Caching jailbreak hidden states...")
    jb_inputs = tokenizer(jailbreak_prompt, return_tensors="pt").to(model.device)
    jb_ids = jb_inputs["input_ids"]

    wrapped = NNsight(model)
    jb_saved = []

    with torch.no_grad():
        with wrapped.trace(jb_ids):
            for i in range(n_layers):
                layer = _get_wrapped_layer(wrapped, i, info)
                h = layer.output[0].save()
                jb_saved.append(h)

    # jb_saved[i] is a tensor of shape (1, seq_len, hidden_dim)
    jb_states = list(jb_saved)

    # Get jailbreak compliance score
    with torch.no_grad():
        jb_out = model(jb_ids)
        jb_score = compliance_score(jb_out.logits[0, -1, :])

    # Get clean baseline score
    clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(model.device)
    clean_ids = clean_inputs["input_ids"]

    with torch.no_grad():
        clean_out = model(clean_ids)
        baseline_score = compliance_score(clean_out.logits[0, -1, :])

    logger.info(f"Baseline (clean): {baseline_score:.4f}, Jailbreak: {jb_score:.4f}")

    # Patch each layer one at a time
    patched_scores = []
    lm_head_proxy = _get_wrapped_lm_head(wrapped, info)
    if lm_head_proxy is None:
        raise RuntimeError("Activation patching requires lm_head, but it could not be located in the model")

    for layer_idx in range(n_layers):
        with torch.no_grad():
            with wrapped.trace(clean_ids):
                layer = _get_wrapped_layer(wrapped, layer_idx, info)
                # Replace the last token position's hidden state at this layer
                # with the jailbreak's last token position hidden state.
                # nnsight 0.6.x returns 2D (seq_len, hidden_dim) here, with
                # the batch dim already squeezed out — so we index with [-1, :]
                # both when reading the cached jb state and when writing the
                # in-trace patch.
                jb_last = _last_pos(jb_states[layer_idx])  # (hidden_dim,)
                layer.output[0][-1, :] = jb_last
                patched_logits = lm_head_proxy.output.save()

        score = compliance_score(_last_pos(patched_logits))
        patched_scores.append(score)

        if (layer_idx + 1) % 10 == 0:
            logger.info(f"Patched {layer_idx + 1}/{n_layers} layers")

    patched_scores = torch.tensor(patched_scores)
    causal_effect = patched_scores - baseline_score

    return {
        "baseline_score": baseline_score,
        "jailbreak_score": jb_score,
        "patched_scores": patched_scores,
        "causal_effect": causal_effect,
        "n_layers": n_layers,
    }


def logit_lens(model, tokenizer, prompt, comply_ids, refuse_ids, top_k=10):
    """Project each layer's hidden state to vocabulary space via the lm_head.

    At each layer, we take the last-position hidden state, apply the final
    layer norm + lm_head, and look at the resulting token probabilities.
    This reveals at which layer the model "decides" to comply vs refuse.

    Args:
        model: The language model (must have lm_head and model.norm).
        tokenizer: The model's tokenizer.
        prompt: Input prompt string.
        comply_ids: Tensor of compliance token IDs.
        refuse_ids: Tensor of refusal token IDs.
        top_k: Number of top predicted tokens to record per layer.

    Returns:
        dict with:
            - comply_probs: tensor (n_layers,) — P(comply tokens) per layer
            - refuse_probs: tensor (n_layers,) — P(refuse tokens) per layer
            - comply_minus_refuse: tensor (n_layers,) — gap per layer
            - top_tokens: list of lists — top-K tokens per layer
            - top_probs: list of lists — corresponding probabilities
            - crossover_layer: int or None — first layer where comply > refuse
            - n_layers: int
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    info = _get_model_info(model)
    n_layers = info["n_layers"]

    comply_ids = comply_ids.to(model.device)
    refuse_ids = refuse_ids.to(model.device)

    # Get the raw norm and lm_head for direct computation (outside trace)
    lm_head = _get_raw_lm_head(model, info)
    if lm_head is None:
        raise RuntimeError("Logit lens requires lm_head, but it could not be located in the model")
    norm = _get_raw_norm(model, info)

    # Cache all hidden states via nnsight
    wrapped = NNsight(model)
    saved = []

    with torch.no_grad():
        with wrapped.trace(input_ids):
            for i in range(n_layers):
                layer = _get_wrapped_layer(wrapped, i, info)
                h = layer.output[0].save()
                saved.append(h)

    comply_probs_list = []
    refuse_probs_list = []
    top_tokens_list = []
    top_probs_list = []

    with torch.no_grad():
        for i in range(n_layers):
            hidden = _last_pos(saved[i]).to(model.device)  # (hidden_dim,)

            # Apply final norm then lm_head (same as the model's output path)
            if norm is not None:
                hidden = norm(hidden.unsqueeze(0)).squeeze(0)
            logits = lm_head(hidden)  # (vocab_size,)

            probs = torch.softmax(logits, dim=-1)

            # Comply/refuse probabilities
            comply_p = probs[comply_ids].sum().item()
            refuse_p = probs[refuse_ids].sum().item()
            comply_probs_list.append(comply_p)
            refuse_probs_list.append(refuse_p)

            # Top-K tokens
            topk = probs.topk(top_k)
            top_tokens_list.append([tokenizer.decode([tid]) for tid in topk.indices])
            top_probs_list.append(topk.values.cpu().tolist())

    comply_probs = torch.tensor(comply_probs_list)
    refuse_probs = torch.tensor(refuse_probs_list)
    gap = comply_probs - refuse_probs

    # Find crossover: first layer where comply > refuse
    crossover = None
    for i in range(n_layers):
        if gap[i] > 0:
            crossover = i
            break

    return {
        "comply_probs": comply_probs,
        "refuse_probs": refuse_probs,
        "comply_minus_refuse": gap,
        "top_tokens": top_tokens_list,
        "top_probs": top_probs_list,
        "crossover_layer": crossover,
        "n_layers": n_layers,
    }
