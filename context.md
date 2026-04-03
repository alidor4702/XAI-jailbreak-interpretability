# Context for Continuation

This file is a handoff briefing. Read this fully before doing anything.

## Who

- **Ali Dor** — MSc AI student at CentraleSupélec. Casual communication style. Uses DCE (not La Ruche) for GPU compute.
- **Elora Drouilhet** — Ali's project partner. Has her own repo: `helloelora/jailbreak-interpretability-slm`. Uses La Ruche (Paris-Saclay HPC, A100 GPUs).
- They work independently on separate repos/clusters but on the same project.

## Rules

- **NEVER** add "Co-Authored-By: Claude" or any AI attribution in git commits.
- **NEVER** assume La Ruche for Ali's work — he uses DCE. Ask him for DCE specifics if you need to write job scripts.
- Ali's GitHub: `alidor4702`. Repo: `alidor4702/xai-jailbreak-interpretability` (private).

## The Project

**Title:** Jailbreak Interpretability in Unified Reasoning Models  
**Course:** Explainable AI (XAI) — CentraleSupélec MSc AI 2025–2026  
**Proposal PDF:** `Docs/PP_AliDor_EloraDrouilhet.pdf` (gitignored, lives locally)

**Goal:** Mechanistically explain WHY safety guardrails in LLMs fail under adversarial jailbreak prompts. Not just finding jailbreaks — using XAI tools to trace the internal "compliance flip" (refusal → obedience).

**Target model:** Mistral Small 3.1 24B (NOT the 119B MoE from the original proposal — they pivoted). Loaded via Unsloth with 4-bit quantization (~14GB VRAM).

## The 4-Stage Pipeline

| Stage | Status | Tool | Location |
|-------|--------|------|----------|
| 1. Model loading | DONE | Unsloth 4-bit | `src/model/loader.py` |
| 2. Prompt fuzzing | DONE | Genetic algorithm | `src/fuzzer/genetic.py`, `seeds.py`, `run.py` |
| 3. Integrated Gradients | NOT STARTED (stub) | Captum | `src/attribution/integrated_gradients.py` |
| 4. Activation tracing | NOT STARTED (stub) | TransformerLens | `src/tracing/activation_analysis.py` |
| Evaluation | NOT STARTED (empty module) | Kendall's Tau + ablation | `src/evaluation/` |

## What the Fuzzer Does

A genetic algorithm that evolves adversarial prompt variants to bypass model guardrails:
- 6 mutation operators: synonym swap, filler insertion, segment rephrasing, context reframing, char substitution (Unicode/leetspeak), token splitting
- Crossover + tournament selection
- Fitness = compliance probability from first-token logits (1 - refusal_prob). Bonus +1.0 if full generation is non-refusal.
- Seeds from HarmBench (cybersecurity + malware categories, 3 each)

## Experiment Results (Elora's runs on La Ruche)

See `results/experiment_01_test_run.md` and `results/experiment_02_full_run.md` for full details.

**Key findings from 600 evaluations:**
- 86.8% overall jailbreak rate
- 3 tiers of guardrail strength: Strong (SQL injection, keylogger, ransomware — refusal_prob 0.93-0.98), Weak (reverse shell, buffer overflow — 0.04-0.31), None (port scanning — 0.03)
- **Best seeds for XAI analysis:** keylogger (malware_0, cleanest flip over 3 gens) and SQL injection (cybersecurity_0)
- Framing mutations ("thriller novel") more effective than obfuscation
- Crossover often destroys prompt meaning → need semantic similarity filter for real jailbreaks

## What Needs to Be Done Next

### 1. Integrated Gradients (Priority — the core XAI contribution)
File: `src/attribution/integrated_gradients.py` (currently `raise NotImplementedError`)

Using Captum's `LayerIntegratedGradients`:
- Take a refused prompt (e.g., keylogger seed) and its jailbroken variant
- Compute token-level attribution for the "compliance flip" — which input tokens most shift the logit gap between refuse and comply
- Compare attribution maps between the two prompts to identify the trigger tokens
- Validate against the fuzzer's known mutations (do IG attributions point at the tokens the fuzzer changed?)

### 2. TransformerLens Activation Tracing
File: `src/tracing/activation_analysis.py` (currently `raise NotImplementedError`)

- Cache internal activations for refused vs. jailbroken prompts
- Compare layer-by-layer (cosine similarity, L2 distance) to find where safety signals diverge
- Identify the specific layers/heads where guardrail suppression happens

### 3. Evaluation
Module: `src/evaluation/` (empty)

- **Kendall's Tau:** Correlation between fuzzer-identified trigger tokens and IG high-attribution tokens (faithfulness metric)
- **Ablation tests:** Mask the circuits identified by tracing → does the model return to refusing?

### 4. HPC Scripts for DCE
The `ruche/` scripts are Elora's. Ali needs equivalent scripts for DCE — but we don't know DCE's specifics yet (partitions, module system, container runtime). Ask Ali before writing these.

## How the Model Loading Works

`src/model/loader.py` — key functions:
- `load_model()` — Unsloth loads `unsloth/Mistral-Small-3.1-24B-Instruct-2503-unsloth-bnb-4bit`, extracts inner `.language_model` (bypasses multimodal wrapper) and `.tokenizer` (from processor)
- `get_first_token_logits()` — forward pass, returns logit vector at last position. Used for compliance scoring.
- `generate()` — greedy autoregressive generation for response text
- `format_chat()` — applies chat template to raw user message

## Repo Structure

```
Project/
├── src/
│   ├── model/loader.py          # Model loading (DONE)
│   ├── fuzzer/genetic.py        # Genetic fuzzer (DONE)
│   ├── fuzzer/seeds.py          # HarmBench seeds (DONE)
│   ├── fuzzer/run.py            # CLI runner (DONE)
│   ├── attribution/integrated_gradients.py  # TODO
│   ├── tracing/activation_analysis.py       # TODO
│   └── evaluation/              # TODO
├── results/                     # Experiment reports
├── ruche/                       # La Ruche HPC scripts (Elora's)
├── notebooks/                   # Empty
├── data/                        # Empty
├── Docs/                        # Proposal PDF (gitignored)
├── requirements.txt
├── README.md
└── context.md                   # This file
```

## Elora's Repo

`https://github.com/helloelora/jailbreak-interpretability-slm`  
Check for updates — she may implement things independently. Her latest commits as of 2026-04-03 were experiment result write-ups and SLURM config tuning (10x10 instead of 20x50).
