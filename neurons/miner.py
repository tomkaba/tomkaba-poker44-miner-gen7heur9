"""Reference Poker44 miner with simple chunk-level behavioral heuristics."""

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, List

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
import poker44.miner_heuristics as _mh
from poker44.miner_heuristics import (
    score_chunk_modern,
    score_chunk_modern_with_route,
    score_chunk_legacy,
    score_chunk_gen7heur5,
    get_chunk_scorer_startup_check,
    chunk_payload_is_legacy,
    _load_ml_model_filtered0,
    _load_ml_model_filtered1,
    _load_ml_model_single_hand,
    get_ml_runtime_stats,
    reset_ml_request_stats,
    filter_other_actions_from_chunk,
)
from poker44.validator.synapse import DetectionSynapse


FORCED_VALIDATOR_HOTKEYS = {
    "5GgnyzhZ6ozkdnQumwRuEaULggvMr2np4SS3N7eDCMMrXoMC",
}

EXTRA_ALLOWED_VALIDATOR_HOTKEYS = {
    "5FZD47WhA1UaVicYAr7pGnWb2YQLMD7uViipDYN2r1AJ5ggD",
}


class Miner(BaseMinerNeuron):
    """Deterministic heuristic miner with dual-mode scoring (legacy + modern chunks)."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 Heuristic Poker44 Miner started")
        
        # Log initialization flags
        ml_max_hands = int(os.getenv("ML_MAX_HANDS", "40"))
        remove_other_flag = os.getenv("REMOVE_OTHER", "0").strip().lower()
        remove_other_enabled = remove_other_flag in ("1", "true", "yes")
        chunk_scorer = "gen7heur5"  # hardcoded for gen7heur5 release
        bt.logging.info(f"[init] ML_MAX_HANDS={ml_max_hands}")
        bt.logging.info(f"[init] REMOVE_OTHER={remove_other_enabled} (raw={remove_other_flag})")
        bt.logging.info("[init] POKER44_CHUNK_SCORER=gen7heur5 (hardcoded for gen7heur5 release)")
        bt.logging.info(
            "[init] Chunk scorer override active: ML_MAX_HANDS routing thresholds are ignored "
            "for per-chunk scoring path."
        )

        # Explicit startup check for chunk-level scorers so operators can catch
        # missing artifacts or loader issues before first validator request.
        scorer_check = get_chunk_scorer_startup_check(chunk_scorer)
        if scorer_check.get("active"):
            details = scorer_check.get("details") or {}
            if scorer_check.get("ok"):
                bt.logging.info(
                    "[init] Chunk scorer startup check: ok "
                    f"scorer={scorer_check.get('scorer')} details={details}"
                )
            else:
                bt.logging.error(
                    "[init] Chunk scorer startup check: FAILED "
                    f"scorer={scorer_check.get('scorer')} "
                    f"error={scorer_check.get('error')} details={details}"
                )
        
        bt.logging.info(f"Axon created: {self.axon}")
        bt.logging.info(f"Build timestamp: {datetime.now(timezone.utc).isoformat()}")
        self._project_root = Path(__file__).resolve().parent.parent
        repo_root = Path(__file__).resolve().parents[1]
        try:
            _git_commit = subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode().strip()
        except Exception:
            _git_commit = os.getenv("POKER44_MODEL_REPO_COMMIT", "")
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[
                Path(__file__).resolve(),
                repo_root / "poker44" / "miner_heuristics.py",
            ],
            defaults={
                "model_name": "poker44_gen7heur5",
                "model_version": "7.5",
                "framework": "python-heuristic",
                "license": "MIT",
                "repo_url": "https://github.com/tomkaba/poker44-miner-gen7heur5",
                "repo_commit": _git_commit,
                "notes": "Gen7heur5 trained on benchmark data from the last 2 days.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained on public human corpus and offline-generated bot chunks. "
                    "No validator-private data used."
                ),
                "training_data_sources": ["hands_generator/human_hands/poker_hands_combined.json.gz"],
                "private_data_attestation": (
                    "This miner does not train on validator-private human data."
                ),
                "data_attestation": (
                    "This miner does not train on validator-private human data."
                ),
            },
        )

        # Optional model selection override for single-hand artifacts.
        # This avoids brittle custom argv handling under bittensor wrappers.
        _mh.configure_single_hand_model_paths_from_env()
        bt.logging.info(
            "Single-hand artifact selection | "
            f"alias={os.getenv('POKER44_SINGLE_HAND_MODEL_ALIAS', '') or 'default(active)'} "
            f"model_path={_mh.ML_SINGLE_HAND_MODEL_PATH} "
            f"scaler_path={_mh.ML_SINGLE_HAND_SCALER_PATH}"
        )

        preload_start = time.perf_counter()
        _load_ml_model_filtered0()
        preload_f0_done = time.perf_counter()
        _load_ml_model_filtered1()
        preload_f1_done = time.perf_counter()
        _load_ml_model_single_hand()
        preload_single_done = time.perf_counter()
        ml_stats = get_ml_runtime_stats()
        bt.logging.info(
            "ML preload timing: "
            f"f0={preload_f0_done - preload_start:.3f}s "
            f"f1={preload_f1_done - preload_f0_done:.3f}s "
            f"single_hand={preload_single_done - preload_f1_done:.3f}s "
            f"total={preload_single_done - preload_start:.3f}s"
        )
        bt.logging.info(
            f"ML preload f0: loaded={ml_stats['ml_model_loaded']} "
            f"available={ml_stats['ml_model_available']} "
            f"attempts={ml_stats['ml_load_attempts']} "
            f"successes={ml_stats['ml_load_successes']} "
            f"model_exists={ml_stats['ml_model_path_exists']} "
            f"scaler_exists={ml_stats['ml_scaler_path_exists']}"
        )
        bt.logging.info(
            f"ML preload f1: loaded={ml_stats['ml_f1_model_loaded']} "
            f"available={ml_stats['ml_f1_model_available']} "
            f"attempts={ml_stats['ml_f1_load_attempts']} "
            f"successes={ml_stats['ml_f1_load_successes']} "
            f"model_exists={ml_stats['ml_f1_model_path_exists']} "
            f"scaler_exists={ml_stats['ml_f1_scaler_path_exists']}"
        )
        bt.logging.info(
            f"ML preload single-hand: loaded={ml_stats['ml_single_hand_model_loaded']} "
            f"available={ml_stats['ml_single_hand_model_available']} "
            f"attempts={ml_stats['ml_single_hand_load_attempts']} "
            f"successes={ml_stats['ml_single_hand_load_successes']} "
            f"model_exists={ml_stats['ml_single_hand_model_path_exists']} "
            f"scaler_exists={ml_stats['ml_single_hand_scaler_path_exists']}"
        )
        _sh_model = _mh._ML_SINGLE_HAND_MODEL
        active_alias = os.getenv("POKER44_SINGLE_HAND_MODEL_ALIAS", "").strip() or "active"
        runtime_model_name = getattr(_sh_model, "_model_name", "") if _sh_model is not None else ""
        runtime_extractor_tag = getattr(_sh_model, "_feature_extractor_tag", "") if _sh_model is not None else ""
        runtime_n_features = getattr(_sh_model, "n_features_in_", "") if _sh_model is not None else ""
        runtime_type = type(_sh_model).__name__ if _sh_model is not None else "None"

        if _sh_model is not None:
            if runtime_model_name:
                self.model_manifest["model_name"] = str(runtime_model_name)
            if runtime_extractor_tag:
                self.model_manifest["model_version"] = str(runtime_extractor_tag)

        self.model_manifest["single_hand_model_alias"] = active_alias
        self.model_manifest["single_hand_model_type"] = runtime_type
        self.model_manifest["single_hand_model_n_features"] = str(runtime_n_features or "unknown")
        self.model_manifest["single_hand_model_extractor_tag"] = str(runtime_extractor_tag or "unknown")
        self.model_manifest["single_hand_model_path"] = str(_mh.ML_SINGLE_HAND_MODEL_PATH)
        self.model_manifest["single_hand_scaler_path"] = str(_mh.ML_SINGLE_HAND_SCALER_PATH)

        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        bt.logging.info(f"Model manifest: {self.model_manifest}")
        bt.logging.info(
            "Manifest source: "
            f"repo_url={self.model_manifest.get('repo_url', '')} "
            f"repo_commit={self.model_manifest.get('repo_commit', '')}"
        )

        bt.logging.info(
            "ML single-hand model identity | "
            f"name={getattr(_sh_model, '_model_name', 'unknown')} "
            f"extractor_tag={getattr(_sh_model, '_feature_extractor_tag', 'unknown')} "
            f"n_features={getattr(_sh_model, 'n_features_in_', 'unknown')} "
            f"type={type(_sh_model).__name__ if _sh_model is not None else 'None'}"
        )
        single_hand_threshold = (
            _mh._single_hand_bot_threshold()
            if hasattr(_mh, "_single_hand_bot_threshold")
            else _mh.SINGLE_HAND_BOT_THRESHOLD
        )
        bt.logging.info(
            "ML single-hand threshold | "
            f"threshold={single_hand_threshold:.3f} "
            f"extractor_tag={getattr(_sh_model, '_feature_extractor_tag', 'unknown')}"
        )
        bt.logging.info(
            "Routing threshold startup | "
            f"ML_MAX_HANDS={_mh.MULTIHAND_SINGLE_HAND_ML_MAX_HANDS} "
            f"policy=len==1->single_hand_ml, "
            f"2<=len<={_mh.MULTIHAND_SINGLE_HAND_ML_MAX_HANDS}->single_hand_ml_vote, "
            f"len>={_mh.MULTIHAND_SINGLE_HAND_ML_MAX_HANDS + 1}->multihand_heuristics"
        )
        bt.logging.debug(
            "Miner ML startup snapshot | "
            f"f0_loaded={ml_stats['ml_model_loaded']} "
            f"f0_path={ml_stats['ml_model_path']} "
            f"f1_loaded={ml_stats['ml_f1_model_loaded']} "
            f"f1_path={ml_stats['ml_f1_model_path']} "
            f"single_hand_loaded={ml_stats['ml_single_hand_model_loaded']} "
            f"single_hand_available={ml_stats['ml_single_hand_model_available']} "
            f"single_hand_path={ml_stats['ml_single_hand_model_path']} "
            f"single_hand_path_exists={ml_stats['ml_single_hand_model_path_exists']}"
        )
        bt.logging.debug(
            "Miner ML startup note | filtered0, filtered1, and single-hand models are preloaded in __init__."
        )

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep tooling available | "
            f"benchmark_doc={repo_root / 'docs' / 'public-benchmark.md'} "
            f"miner_doc={repo_root / 'docs' / 'miner.md'} "
            f"anti_leakage_doc={repo_root / 'docs' / 'anti-leakage.md'}"
        )
        bt.logging.info(
            "Public benchmark command: "
            "python scripts/publish/publish_public_benchmark.py --skip-wandb"
        )
        bt.logging.info(
            "Purpose: train, validate and refine miner models against the public benchmark "
            "while Poker44 moves toward more dynamic evaluation."
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one deterministic bot-risk score per chunk."""
        reset_ml_request_stats()
        chunks: List[List[dict]] = synapse.chunks or []

        def _preview(seq, limit=8):
            if len(seq) <= limit:
                return seq
            return [*seq[:limit], "..."]

        chunk_sizes = [len(chunk or []) for chunk in chunks]
        bt.logging.debug(f"[miner] Received {len(chunks)} chunk(s); first sizes={_preview(chunk_sizes)}")

        # Check if REMOVE_OTHER flag is enabled
        remove_other_flag = os.getenv("REMOVE_OTHER", "0").strip().lower()
        remove_other_enabled = remove_other_flag in ("1", "true", "yes")
        
        if remove_other_enabled:
            filtered_chunks = []
            total_removed = 0
            for idx, chunk in enumerate(chunks):
                filtered_chunk, removed_count = filter_other_actions_from_chunk(chunk)
                filtered_chunks.append(filtered_chunk)
                total_removed += removed_count
                if removed_count > 0:
                    bt.logging.debug(f"[miner] chunk[{idx}] removing {removed_count} 'other' entries")
            
            chunks = filtered_chunks
            if total_removed > 0:
                bt.logging.debug(f"[miner] Total 'other' entries removed: {total_removed} across {len(chunks)} chunk(s)")
            bt.logging.info(f"[REMOVE_OTHER] filter ran: removed={total_removed} chunks={len(chunks)}")
            
            chunk_sizes = [len(chunk or []) for chunk in chunks]

        chunk_modes = ["legacy" if chunk_payload_is_legacy(chunk) else "modern" for chunk in chunks]
        processing_start = time.perf_counter()
        scores = []
        chunk_routes = []

        # gen7heur5 scorer bypasses per-hand ML models entirely
        chunk_scorer = "gen7heur5"  # hardcoded for gen7heur5 release

        if chunk_scorer == "gen7heur5":
            for index, (chunk, mode) in enumerate(zip(chunks, chunk_modes)):
                if mode == "legacy":
                    score = score_chunk_legacy(chunk)
                    route = "legacy_payload_heuristics"
                else:
                    score, route = score_chunk_gen7heur5(chunk)
                scores.append(score)
                chunk_routes.append(route)
                bt.logging.debug(
                    f"[miner] chunk[{index}] size={len(chunk or [])} "
                    f"mode={mode} route={route} score={float(score):.6f}"
                )
        else:
            for index, (chunk, mode) in enumerate(zip(chunks, chunk_modes)):
                if mode == "legacy":
                    score = score_chunk_legacy(chunk)
                    route = "legacy_payload_heuristics"
                else:
                    score, route = score_chunk_modern_with_route(chunk)
                scores.append(score)
                chunk_routes.append(route)
                bt.logging.debug(
                    f"[miner] chunk[{index}] size={len(chunk or [])} "
                    f"mode={mode} route={route} score={float(score):.6f}"
                )
        processing_elapsed = time.perf_counter() - processing_start
        avg_ms = (processing_elapsed / max(1, len(chunks))) * 1000.0
        bt.logging.debug(
            "[miner] batch processing timing: "
            f"chunks={len(chunks)} total_seconds={processing_elapsed:.6f} "
            f"avg_ms_per_chunk={avg_ms:.3f}"
        )
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        
        # DEBUG: Log synapse state before response
        bt.logging.debug(
            f"[DEBUG] Before sending: synapse.risk_scores={synapse.risk_scores}, "
            f"type={type(synapse.risk_scores)}, len={len(synapse.risk_scores) if synapse.risk_scores else 'None'}"
        )
        
        bt.logging.debug(
            f"[miner] Responding with scores={_preview(scores)} "
            f"predictions={_preview(synapse.predictions)}"
        )
        bt.logging.info(f"Miner Predictions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks with heuristic risks.")

        ml_stats = get_ml_runtime_stats()
        modern_chunks = chunk_modes.count("modern")
        modern_chunk_sizes = [len(chunk or []) for chunk, mode in zip(chunks, chunk_modes) if mode == "modern"]
        modern_single_hand_chunks = sum(1 for size in modern_chunk_sizes if size == 1)
        ml_max_hands = int(getattr(_mh, "MULTIHAND_SINGLE_HAND_ML_MAX_HANDS", 40))
        modern_small_multihand_chunks = sum(1 for size in modern_chunk_sizes if 2 <= size <= ml_max_hands)
        modern_long_multihand_chunks = sum(1 for size in modern_chunk_sizes if size >= (ml_max_hands + 1))
        accounted_modern = (
            int(ml_stats["request_f0_ml_used"])
            + int(ml_stats["request_f0_heur_fallback"])
            + int(ml_stats["request_f1_ml_used"])
            + int(ml_stats["request_f1_heur_fallback"])
            + int(ml_stats.get("request_f1_hardcut_forced_human", 0))
            + int(ml_stats.get("request_f2plus_forced_human", 0))
            + int(ml_stats.get("request_multihand_small_ml_used", 0))
            + int(ml_stats.get("request_multihand_small_heur_fallback", 0))
        )
        bt.logging.debug(
            "Routing policy (modern chunks): "
            "len==1 -> single_hand_ml; "
            f"2<=len<={ml_max_hands} -> single_hand_ml_vote; "
            f"len>={ml_max_hands + 1} -> legacy_multihand_heuristics"
        )
        bt.logging.debug(
            "Routing counts (modern chunks): "
            f"single_hand={modern_single_hand_chunks} "
            f"small_multihand_2_{ml_max_hands}={modern_small_multihand_chunks} "
            f"long_multihand_{ml_max_hands + 1}_plus={modern_long_multihand_chunks}"
        )
        bt.logging.info(
            f"ML runtime: modern_chunks={modern_chunks} "
            f"f0_ml_used={ml_stats['request_f0_ml_used']} "
            f"f0_heur_fallback={ml_stats['request_f0_heur_fallback']} "
            f"f1_ml_used={ml_stats['request_f1_ml_used']} "
            f"f1_heur_fallback={ml_stats['request_f1_heur_fallback']} "
            f"f1_hardcut_forced_human={ml_stats.get('request_f1_hardcut_forced_human', 0)} "
            f"f2plus_forced_human={ml_stats.get('request_f2plus_forced_human', 0)} "
            f"small_multihand_ml_used={ml_stats.get('request_multihand_small_ml_used', 0)} "
            f"small_multihand_heur_fallback={ml_stats.get('request_multihand_small_heur_fallback', 0)} "
            f"accounted_modern={accounted_modern} "
            f"f0_loaded={ml_stats['ml_model_loaded']} "
            f"f1_loaded={ml_stats['ml_f1_model_loaded']}"
        )

        source_hotkey = getattr(getattr(synapse, "dendrite", None), "hotkey", "unknown")
        self._append_request_log(
            validator_hotkey=source_hotkey,
            chunk_sizes=chunk_sizes,
            chunk_modes=chunk_modes,
            scores=scores,
            chunk_routes=chunk_routes,
            predictions=synapse.predictions,
            chunks=chunks,
            ml_stats=ml_stats,
        )

        return synapse

    @staticmethod
    def _flag_enabled(config_section, attr, default=None):
        value = getattr(config_section, attr, default) if config_section else default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "on"}
        return bool(value)

    def _allowed_validator_hotkeys(self) -> set[str]:
        cfg = getattr(self.config, "blacklist", None)
        allowed = set(FORCED_VALIDATOR_HOTKEYS) | set(EXTRA_ALLOWED_VALIDATOR_HOTKEYS)

        def _normalize(value) -> set[str]:
            if value is None:
                return set()
            if isinstance(value, (list, tuple, set)):
                iterable = value
            else:
                iterable = str(value).split(",")
            return {str(item).strip() for item in iterable if str(item).strip()}

        allowed |= _normalize(getattr(cfg, "forced_validator_hotkey", None))
        allowed |= _normalize(getattr(cfg, "forced_validator_hotkeys", None))
        allowed |= _normalize(getattr(cfg, "extra_validator_hotkeys", None))
        return allowed

    def score_chunk(self, chunk: list[dict]) -> float:
        return score_chunk_legacy(chunk) if chunk_payload_is_legacy(chunk) else score_chunk_modern(chunk)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        allow_non_registered = self._flag_enabled(
            getattr(self.config, "blacklist", None),
            "allow_non_registered",
            False,
        )
        force_validator_permit = self._flag_enabled(
            getattr(self.config, "blacklist", None),
            "force_validator_permit",
            True,
        )
        allowed_hotkeys = self._allowed_validator_hotkeys()

        if synapse.dendrite.hotkey in allowed_hotkeys:
            bt.logging.debug(f"Allowing validator hotkey {synapse.dendrite.hotkey}")
            return False, "Validator allowlist"

        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            if not allow_non_registered:
                bt.logging.trace(f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}")
                return True, "Unrecognized hotkey"
            bt.logging.debug(
                f"Allowing unregistered hotkey {synapse.dendrite.hotkey} "
                "because allow_non_registered=True"
            )
            return False, "Unregistered hotkey allowed"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        if force_validator_permit and not self.metagraph.validator_permit[uid]:
            bt.logging.warning(f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}")
            return True, "Non-validator hotkey"

        bt.logging.trace(f"Not blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        return False, "Hotkey recognized!"

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)

    def _get_log_path(self) -> Path:
        uid = getattr(self, "uid", None)
        suffix = uid if uid is not None else "unknown"
        return self._project_root / f"miner_{suffix}.log"

    def _full_logging_enabled(self) -> bool:
        cfg = getattr(self.config, "logging", None)
        config_flag = getattr(cfg, "disable_full_logs", False)
        env_flag = os.getenv("POKER44_DISABLE_FULL_LOGS", "false").strip().lower()
        env_disable = env_flag in {"1", "true", "yes", "on"}
        return not (config_flag or env_disable)

    def _append_request_log(
        self,
        validator_hotkey,
        chunk_sizes,
        chunk_modes,
        scores,
        chunk_routes,
        predictions,
        chunks,
        ml_stats,
    ) -> None:
        if not self._full_logging_enabled():
            bt.logging.debug("Full request logging disabled; skipping miner log entry.")
            return
        entry = {
            "timestamp": time.time(),
            "validator_hotkey": validator_hotkey,
            "miner_hotkey": getattr(self.wallet.hotkey, "ss58_address", "unknown"),
            "chunk_count": len(chunk_sizes),
            "chunk_sizes": chunk_sizes,
            "chunk_modes": chunk_modes,
            "chunk_routes": chunk_routes,
            "scores": scores,
            "predictions": predictions,
            "ml_runtime": ml_stats,
            "chunks": chunks,
        }
        try:
            log_path = self._get_log_path()
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as log_error:
            bt.logging.warning(f"Failed to append miner request log: {log_error}")

    def _dump_request_payload(self, *args, **kwargs):
        return


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Random miner running...")
        while True:
            profile_alias = str(
                (getattr(miner, "model_manifest", {}) or {}).get("single_hand_model_alias")
                or os.getenv("POKER44_SINGLE_HAND_MODEL_ALIAS", "")
                or "active"
            )
            chunk_scorer = os.getenv("POKER44_CHUNK_SCORER", "").strip().lower() or "default_ml"
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]} "
                f"| Profile: {profile_alias} | Scorer: {chunk_scorer}"
            )
            time.sleep(5 * 60)
