# Jailbreak Interpretability in Unified Reasoning Models

**Authors:** Ali Dor & Elora Drouilhet  
**Course:** Explainable Artificial Intelligence — CentraleSupélec MSc AI (2025–2026)

---

## Overview

Safety-aligned LLMs can be tricked into producing harmful content by adversarial "jailbreak" prompts. This project goes beyond detecting jailbreaks — we mechanistically explain **where and how** the compliance flip happens inside the model.

We combine a **genetic fuzzer** that discovers jailbreak prompts with a suite of **XAI tools** (Integrated Gradients, activation patching, logit lens, layer divergence) to pinpoint the exact tokens and transformer layers responsible for suppressing safety behavior.

**Target model:** [Mistral Small 3.1 24B](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) — loaded via [Unsloth](https://github.com/unslothai/unsloth) in 4-bit quantization (~14 GB VRAM).

---

## Key Findings

### Fuzzer results
- **6/6 seed prompts** successfully jailbroken across cybersecurity and malware categories
- **166 validated jailbreaks** after HarmBench classification and semantic filtering (17 false positives removed)
- Mean HarmBench score: **0.99** across kept jailbreaks

### XAI analysis (10 jailbreak pairs)

| Metric | Value |
|--------|-------|
| Compliance shift (Δ) | **+0.94 to +5.06** across examples |
| Safety decision layers | **L17–L24** (43.1% of total causal importance) |
| Late commitment layers | **L34–L39** (model only commits to refuse/comply in final 5 layers) |
| Divergence–causation correlation | Pearson **r = 0.95** (example 0) |
| Single-layer recovery | Patching **1 layer** often matches or exceeds the full jailbreak effect |

### Core insight

> Jailbreaks **attenuate** the refusal signal in layers 17–24 rather than injecting a new "comply" signal. The model's safety circuitry is not bypassed — it is gradually weakened until the narrowed margin flips the output in the final layers.

---

## Sample Results

### Token Attribution — Which words drive compliance?

| Clean prompt (refused) | Jailbreak prompt (complied) |
|---|---|
| ![Clean tokens](outputs/fuzzer_170260/xai_analysis_2/example_0_tokens_clean.png) | ![Jailbreak tokens](outputs/fuzzer_170260/xai_analysis_2/example_0_tokens_jailbreak.png) |

Blue = pushes toward refusal, Red = pushes toward compliance. In the jailbreak, roleplay tokens ("Imagine", "protagonist", "without restrictions") override the danger signal from "SQL injection".

### 5-Panel Dashboard

![Dashboard](outputs/fuzzer_170260/xai_analysis_2/example_0_dashboard.png)

**(A)** Top token attributions compared, **(B)** Layer divergence between clean and jailbreak hidden states, **(C)** Causal effect per layer via activation patching, **(D)** Ablation test validating IG findings, **(E)** Divergence vs causal effect scatter (r=0.95).

### Cross-Seed Layer Importance

![Layer importance](outputs/fuzzer_170260/xai_analysis_2/cross_seed_layer_importance.png)

Aggregated across all 10 examples: layers 21–24 consistently emerge as the most causally important for the compliance flip.

### Logit Lens — When does the model decide?

![Logit lens comparison](outputs/fuzzer_170260/xai_analysis_2/example_0_logit_lens_comparison.png)

Left: clean prompt — strong refusal signal (blue) emerges at layer 34. Right: jailbreak — refusal signal is attenuated, compliance gap shrinks from −1.0 to −0.35.

---

## Pipeline

```
Seed Prompts ──► Genetic Fuzzer ──► Jailbreak Prompts ──► HarmBench Validation
                                                                  │
                                                                  ▼
                                              ┌─────────────────────────────────┐
                                              │      XAI Analysis Pipeline      │
                                              ├─────────────────────────────────┤
                                              │ 1. Integrated Gradients (Captum)│
                                              │    → Token-level attribution    │
                                              │                                 │
                                              │ 2. Activation Patching (nnsight)│
                                              │    → Causal layer identification│
                                              │                                 │
                                              │ 3. Logit Lens                   │
                                              │    → Per-layer decision tracking │
                                              │                                 │
                                              │ 4. Layer Divergence             │
                                              │    → Representation differences │
                                              │                                 │
                                              │ 5. Ablation Test                │
                                              │    → Attribution validation     │
                                              └─────────────────────────────────┘
                                                                  │
                                                                  ▼
                                                        Visualizations &
                                                        Cross-Seed Analysis
```

### Analysis techniques explained

| Technique | What it measures | Method |
|-----------|-----------------|--------|
| **Integrated Gradients** | Which input tokens drive the comply/refuse decision | Captum `LayerIntegratedGradients` on the embedding layer; attributions interpolated from zero baseline |
| **Activation Patching** | Which layers causally control the compliance flip | For each layer, swap the jailbreak hidden state into the clean forward pass and measure compliance shift |
| **Logit Lens** | At which layer the model commits to its decision | Project each layer's hidden state through the final norm + LM head to get per-layer vocabulary predictions |
| **Layer Divergence** | Where clean and jailbreak representations differ | Cosine distance between hidden states at each layer (observational, not causal) |
| **Ablation Test** | Whether IG-identified tokens truly matter | Progressively mask top-K attributed tokens and measure compliance score drop |

---

## Project Structure

```
├── src/
│   ├── model/              # Unsloth model loading & inference
│   │   └── loader.py
│   ├── fuzzer/             # Genetic algorithm prompt fuzzer
│   │   ├── genetic.py      # Core GA: mutation, crossover, selection
│   │   ├── seeds.py        # Seed prompt definitions
│   │   ├── run.py          # Fuzzer entry point
│   │   ├── validator.py    # Semantic jailbreak validator
│   │   └── harmbench_judge.py  # HarmBench LLM classifier
│   ├── attribution/        # Feature attribution
│   │   └── integrated_gradients.py  # Captum IG with user-span detection
│   ├── tracing/            # Mechanistic interpretability
│   │   └── activation_analysis.py   # nnsight-based patching, logit lens, divergence
│   └── evaluation/         # Evaluation metrics
│       └── metrics.py
├── scripts/
│   ├── run_xai_analysis.py # End-to-end XAI pipeline
│   ├── plotting.py         # All visualization code
│   ├── harmbench_validate.py
│   └── harmbench_revise.py
├── dce/                    # DCE cluster SLURM scripts
├── outputs/
│   └── fuzzer_170260/
│       ├── harmbench/          # HarmBench classification results
│       ├── harmbench_revised/  # Semantically validated results
│       ├── xai_analysis/       # Initial XAI run (35 files)
│       └── xai_analysis_2/     # Refined XAI run (65 files, 10 examples)
└── data/                   # HarmBench & JailbreakBench datasets
```

---

## Output Files Per Example

Each analyzed jailbreak pair produces 6 figures:

| File | Description |
|------|-------------|
| `tokens_clean.png` | Token attribution heatmap for the clean (refused) prompt |
| `tokens_jailbreak.png` | Token attribution heatmap for the jailbreak (complied) prompt |
| `dashboard.png` | 5-panel analysis: IG comparison, divergence, patching, ablation, correlation |
| `logit_lens_clean.png` | Per-layer P(comply) vs P(refuse) for clean prompt |
| `logit_lens_jailbreak.png` | Per-layer P(comply) vs P(refuse) for jailbreak prompt |
| `logit_lens_comparison.png` | Side-by-side logit lens comparison |

Plus 4 cross-seed aggregation figures:

| File | Description |
|------|-------------|
| `compliance_landscape.png` | Clean vs jailbreak vs best-single-layer-patch scores per example |
| `cross_seed_divergence.png` | Divergence + causal effect heatmaps across all examples |
| `cross_seed_layer_importance.png` | Aggregate causal importance per layer |
| `cross_seed_logit_lens.png` | Logit lens compliance gap across all examples |

---

## Setup

```bash
# Clone and install
git clone https://github.com/alidor4702/XAI-jailbreak-interpretability.git
cd XAI-jailbreak-interpretability
pip install -r requirements.txt

# Run the fuzzer (requires GPU)
python -m src.fuzzer.run

# Run XAI analysis on fuzzer results
python -m scripts.run_xai_analysis outputs/fuzzer_170260 --n-per-seed 2 --out-subdir xai_analysis
```

### Requirements

- Python 3.10+
- CUDA GPU with ≥24 GB VRAM (RTX 3090 / A100)
- Key dependencies: `unsloth`, `nnsight`, `captum`, `transformers`, `torch`

---

## References

1. Mistral AI Team. (2025). *Mistral Small 3.1.* [[Model Card]](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)
2. Mazeika, M., et al. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming.* ICML.
3. Fiotto-Kaufman, J., et al. (2024). *NNsight and NDIF: Democratizing Access to Foundation Model Internals.* ICML.
4. Sundararajan, M., et al. (2017). *Axiomatic Attribution for Deep Networks.* ICML.
5. nostalgebraist. (2020). *interpreting GPT: the logit lens.* LessWrong.
6. Wei, A., et al. (2024). *Jailbroken: How Does LLM Safety Training Fail?* NeurIPS.
7. Unit 42. (2026). *Open, Closed and Broken: Prompt Fuzzing Finds LLMs Still Fragile.* Palo Alto Networks.
