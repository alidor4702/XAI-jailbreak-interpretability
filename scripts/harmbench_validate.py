"""
Validate fuzzer results using the HarmBench gold-standard classifier.

Loads the HarmBench-Llama-2-13b-cls model, runs it on all candidates from
a fuzzer run, and writes results with harmbench_jailbroken + harmbench_score.

Usage:
    python3 -m scripts.harmbench_validate outputs/fuzzer_170260

Output goes to: outputs/fuzzer_170260/harmbench/
"""

import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fuzzer.harmbench_judge import HarmBenchJudge
from src.fuzzer.seeds import get_all_seeds


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m scripts.harmbench_validate <run_dir>")
        sys.exit(1)

    run_dir = sys.argv[1]

    seeds = get_all_seeds()
    seed_map = {i: (cat, prompt) for i, (cat, prompt) in enumerate(seeds)}

    out_dir = os.path.join(run_dir, "harmbench")
    os.makedirs(out_dir, exist_ok=True)

    # Load judge
    judge = HarmBenchJudge()

    total_old_jb = 0
    total_hb_jb = 0
    total_candidates = 0

    for fname in sorted(os.listdir(run_dir)):
        if not fname.startswith("seed_"):
            continue

        parts = fname.split("_")
        seed_idx = int(parts[2])
        category, seed_prompt = seed_map[seed_idx]

        path = os.path.join(run_dir, fname)
        with open(path) as f:
            data = json.load(f)

        candidates = data.get("candidates", [])
        if not candidates:
            logger.warning(f"No candidates in {fname}, skipping")
            continue

        logger.info(f"=== {fname} ({len(candidates)} candidates) ===")
        logger.info(f"  Seed: {seed_prompt}")

        # Prepare behaviors and generations for batch judging
        behaviors = [seed_prompt] * len(candidates)
        generations = [c["response"] for c in candidates]

        start = time.time()
        results = judge.judge_batch(behaviors, generations)
        elapsed = time.time() - start

        # Annotate candidates
        old_jb_count = 0
        hb_jb_count = 0
        for c, (is_jb, score) in zip(candidates, results):
            c["harmbench_jailbroken"] = is_jb
            c["harmbench_score"] = round(score, 4)
            if c.get("jailbroken", False):
                old_jb_count += 1
            if is_jb:
                hb_jb_count += 1

        # Rebuild jailbreaks list based on HarmBench
        hb_jailbreaks = [c for c in candidates if c["harmbench_jailbroken"]]
        hb_jailbreaks.sort(key=lambda c: c["harmbench_score"], reverse=True)

        data["candidates"] = candidates
        data["jailbreaks_harmbench"] = hb_jailbreaks
        data["harmbench_stats"] = {
            "total_candidates": len(candidates),
            "old_jailbreaks": old_jb_count,
            "harmbench_jailbreaks": hb_jb_count,
            "elapsed_seconds": round(elapsed, 1),
        }

        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        total_old_jb += old_jb_count
        total_hb_jb += hb_jb_count
        total_candidates += len(candidates)

        logger.info(
            f"  Old jailbreaks: {old_jb_count}, "
            f"HarmBench jailbreaks: {hb_jb_count}, "
            f"time: {elapsed:.1f}s"
        )

    logger.info("=" * 60)
    logger.info("HARMBENCH VALIDATION SUMMARY")
    logger.info(f"Total candidates judged:  {total_candidates}")
    logger.info(f"Old jailbreaks (naive):   {total_old_jb}")
    logger.info(f"HarmBench jailbreaks:     {total_hb_jb}")
    logger.info(f"Results saved to:         {out_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
