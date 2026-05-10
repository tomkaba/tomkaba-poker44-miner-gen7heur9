"""Custom public benchmark generator with selectable bot profile families.

This module intentionally builds on top of the existing public benchmark flow and
only patches the default bot profile source during dataset generation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from hands_generator import mixed_dataset_provider as mdp
from hands_generator.data_generator import _default_bot_profiles
from hands_generator.public_benchmark import (
    DEFAULT_PUBLIC_BENCHMARK_PATH,
    DEFAULT_HUMAN_JSON_PATH,
    PublicBenchmarkConfig,
    _compute_payload_hash,
    build_public_benchmark,
    save_public_benchmark,
)
from hands_generator.bot_hands.generate_poker_data import BotProfile


PROFILE_ALIASES = {
    "balanced": "balanced",
    "tag": "tight_aggressive",
    "tight_aggressive": "tight_aggressive",
    "lag": "loose_aggressive",
    "loose_aggressive": "loose_aggressive",
    "tp": "tight_passive",
    "tight_passive": "tight_passive",
    "lp": "loose_passive",
    "loose_passive": "loose_passive",
}


@dataclass
class CustomPublicBenchmarkConfig:
    human_json_path: Path = DEFAULT_HUMAN_JSON_PATH
    output_path: Path = DEFAULT_PUBLIC_BENCHMARK_PATH
    chunk_count: int = 40
    min_hands_per_chunk: int = 60
    max_hands_per_chunk: int = 120
    human_ratio: float = 0.5
    seed: int = 44
    validation_ratio: float = 0.25
    bot_profile_preset: str = "default_mix"
    bot_profiles: str = ""
    progress_every: int = 500


def available_bot_profiles() -> List[str]:
    return [profile.name for profile in _default_bot_profiles()]


def available_profile_presets() -> Dict[str, List[str]]:
    return {
        "default_mix": available_bot_profiles(),
        "balanced_only": ["balanced"],
        "tight_aggressive_only": ["tight_aggressive"],
        "loose_aggressive_only": ["loose_aggressive"],
        "tight_passive_only": ["tight_passive"],
        "loose_passive_only": ["loose_passive"],
        "aggressive_mix": ["tight_aggressive", "loose_aggressive"],
        "passive_mix": ["tight_passive", "loose_passive"],
        "tight_mix": ["tight_aggressive", "tight_passive"],
        "loose_mix": ["loose_aggressive", "loose_passive"],
        "no_balanced": [
            "tight_aggressive",
            "loose_aggressive",
            "tight_passive",
            "loose_passive",
        ],
    }


def _fresh_default_profiles() -> Dict[str, BotProfile]:
    return {profile.name: profile for profile in _default_bot_profiles()}


def _normalize_profile_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        return normalized
    return PROFILE_ALIASES.get(normalized, normalized)


def _parse_profile_names(text: str) -> List[str]:
    names = [_normalize_profile_name(token) for token in text.split(",") if token.strip()]
    if not names:
        raise ValueError("bot profile list is empty")
    return names


def resolve_bot_profiles(preset: str, explicit_profiles: str = "") -> Tuple[str, List[BotProfile]]:
    base_profiles = _fresh_default_profiles()
    presets = available_profile_presets()

    if explicit_profiles.strip():
        names = _parse_profile_names(explicit_profiles)
        resolved_preset = "custom"
    else:
        resolved_preset = preset or "default_mix"
        if resolved_preset not in presets:
            raise ValueError(
                f"Unknown bot profile preset: {resolved_preset}. Available: {', '.join(sorted(presets))}"
            )
        names = presets[resolved_preset]

    missing = [name for name in names if name not in base_profiles]
    if missing:
        raise ValueError(
            f"Unknown bot profile(s): {', '.join(missing)}. Available: {', '.join(sorted(base_profiles))}"
        )

    profiles = [base_profiles[name] for name in names]
    return resolved_preset, profiles


@contextmanager
def patched_default_bot_profiles(profiles: Iterable[BotProfile]):
    original = mdp._default_bot_profiles
    frozen = list(profiles)

    def _patched() -> List[BotProfile]:
        return [BotProfile(**asdict(profile)) for profile in frozen]

    mdp._default_bot_profiles = _patched
    try:
        yield
    finally:
        mdp._default_bot_profiles = original


@contextmanager
def patched_bot_generation_progress(progress_every: int):
    original = mdp._build_bot_chunks

    if progress_every <= 0:
        yield
        return

    def _patched(*, bot_sizes, bot_profiles, human_pool, human_signatures, human_structures, rng, candidate_attempts):
        total = len(bot_sizes)
        generated = []
        for idx, size in enumerate(bot_sizes, 1):
            # Keep target-signature progression equivalent to original implementation.
            target_sig = [human_signatures[(idx - 1) % len(human_signatures)]]
            target_struct = [human_structures[(idx - 1) % len(human_structures)]]
            generated.extend(
                original(
                    bot_sizes=[size],
                    bot_profiles=bot_profiles,
                    human_pool=human_pool,
                    human_signatures=target_sig,
                    human_structures=target_struct,
                    rng=rng,
                    candidate_attempts=candidate_attempts,
                )
            )
            if idx % progress_every == 0 or idx == total:
                print(
                    f"[bot-gen] generated {idx}/{total} bot chunks",
                    flush=True,
                )
        return generated

    mdp._build_bot_chunks = _patched
    try:
        yield
    finally:
        mdp._build_bot_chunks = original


def build_public_benchmark_custom(cfg: CustomPublicBenchmarkConfig):
    resolved_preset, profiles = resolve_bot_profiles(cfg.bot_profile_preset, cfg.bot_profiles)
    base_cfg = PublicBenchmarkConfig(
        human_json_path=cfg.human_json_path,
        output_path=cfg.output_path,
        chunk_count=cfg.chunk_count,
        min_hands_per_chunk=cfg.min_hands_per_chunk,
        max_hands_per_chunk=cfg.max_hands_per_chunk,
        human_ratio=cfg.human_ratio,
        seed=cfg.seed,
        validation_ratio=cfg.validation_ratio,
    )

    with patched_default_bot_profiles(profiles):
        with patched_bot_generation_progress(int(cfg.progress_every)):
            payload, _ = build_public_benchmark(base_cfg)

    payload.pop("dataset_hash", None)
    payload["config"]["bot_profile_preset"] = resolved_preset
    payload["config"]["bot_profiles"] = [profile.name for profile in profiles]
    payload["config"]["bot_profile_count"] = len(profiles)
    payload["stats"]["bot_profile_preset"] = resolved_preset
    payload["stats"]["bot_profile_count"] = len(profiles)
    dataset_hash = _compute_payload_hash(payload)
    payload["dataset_hash"] = dataset_hash
    return payload, dataset_hash


__all__ = [
    "DEFAULT_PUBLIC_BENCHMARK_PATH",
    "DEFAULT_HUMAN_JSON_PATH",
    "CustomPublicBenchmarkConfig",
    "available_bot_profiles",
    "available_profile_presets",
    "build_public_benchmark_custom",
    "resolve_bot_profiles",
    "save_public_benchmark",
]