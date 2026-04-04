"""
Re-annotate existing fuzzer results with the semantic jailbreak validator.

Reads the original per-seed JSON files, applies validate_jailbreak() to every
candidate, and writes updated files with:
  - jailbroken: new validated boolean
  - jailbroken_old: original naive boolean
  - validation_reason: why the validator accepted/rejected

Usage:
    python3 -m scripts.reannotate_results outputs/fuzzer_170260
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fuzzer.validator import validate_jailbreak
from src.fuzzer.seeds import get_all_seeds


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m scripts.reannotate_results <output_dir>")
        sys.exit(1)

    run_dir = sys.argv[1]
    seeds = get_all_seeds()
    seed_map = {i: prompt for i, (cat, prompt) in enumerate(seeds)}

    out_dir = os.path.join(run_dir, "validated")
    os.makedirs(out_dir, exist_ok=True)

    total_old = 0
    total_new = 0

    for fname in sorted(os.listdir(run_dir)):
        if not fname.startswith("seed_"):
            continue

        parts = fname.split("_")
        seed_idx = int(parts[2])
        seed_prompt = seed_map[seed_idx]

        path = os.path.join(run_dir, fname)
        with open(path) as f:
            data = json.load(f)

        # Re-annotate all candidates
        for cand in data.get("candidates", []):
            old_jb = cand.get("jailbroken", False)
            is_jb, reason = validate_jailbreak(seed_prompt, cand["prompt"], cand["response"])
            cand["jailbroken_old"] = old_jb
            cand["jailbroken"] = is_jb
            cand["validation_reason"] = reason

        # Rebuild jailbreaks list
        old_jailbreaks = data.get("jailbreaks", [])
        new_jailbreaks = [c for c in data["candidates"] if c["jailbroken"]]
        new_jailbreaks.sort(key=lambda c: c["score"], reverse=True)
        data["jailbreaks"] = new_jailbreaks
        data["jailbreaks_old_count"] = len(old_jailbreaks)

        total_old += len(old_jailbreaks)
        total_new += len(new_jailbreaks)

        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        print(
            f"{fname}: {len(old_jailbreaks)} -> {len(new_jailbreaks)} "
            f"(rejected {len(old_jailbreaks) - len(new_jailbreaks)})"
        )

    print(f"\nTotal: {total_old} -> {total_new} (rejected {total_old - total_new})")
    print(f"Validated results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
