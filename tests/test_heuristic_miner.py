from pathlib import Path
import unittest

from hands_generator.mixed_dataset_provider import (
    DEFAULT_HUMAN_JSON_PATH,
    MixedDatasetConfig,
    build_mixed_labeled_chunks,
)
from poker44.miner_heuristics import score_chunk


class HeuristicMinerTest(unittest.TestCase):
    def test_heuristic_miner_matches_labels(self) -> None:
        """Ensure the heuristic miner remains perfectly aligned with generator labels."""

        tmp_path = Path("data/test_outputs")
        tmp_path.mkdir(parents=True, exist_ok=True)
        cfg = MixedDatasetConfig(
            human_json_path=DEFAULT_HUMAN_JSON_PATH,
            output_path=tmp_path / "validator_mixed_chunks.json",
            chunk_count=16,
            min_hands_per_chunk=60,
            max_hands_per_chunk=70,
            human_ratio=0.5,
            refresh_seconds=60,
            seed=20260310,
            validator_secret_key="heuristic-check",
            bot_candidate_attempts_per_chunk=4,
            max_bot_generation_rounds=4,
            max_shortcut_rule_accuracy=0.70,
        )

        chunks, dataset_hash, stats = build_mixed_labeled_chunks(cfg, window_id=1)

        mismatches = []
        for idx, chunk in enumerate(chunks):
            hands = chunk.get("hands", [])
            risk_score = score_chunk(hands)
            predicted_bot = risk_score >= 0.5
            expected_bot = bool(chunk.get("is_bot", False))
            if predicted_bot != expected_bot:
                mismatches.append(
                    {
                        "index": idx,
                        "size": len(hands),
                        "score": risk_score,
                        "expected": expected_bot,
                    }
                )

        self.assertFalse(
            mismatches,
            msg=(
                "Heuristic miner diverged from generator labels. "
                f"Dataset={dataset_hash} Stats={stats} Mismatches={mismatches[:3]}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
