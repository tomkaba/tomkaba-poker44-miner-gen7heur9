# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Poker44 validator entrypoint wired into the base Bittensor neuron."""
# neuron/validator.py

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import bittensor as bt
from dotenv import load_dotenv

from poker44 import __version__
from poker44.base.validator import BaseValidatorNeuron
from poker44.utils.config import config
from poker44.utils.wandb_helper import ValidatorWandbHelper
from poker44.validator.forward import forward as forward_cycle
from poker44.validator.integrity import (
    load_json_registry,
    normalize_uid_key_registry,
)
from hands_generator.mixed_dataset_provider import (
    DEFAULT_OUTPUT_PATH,
    MixedDatasetConfig,
    TimedMixedDatasetProvider,
)

load_dotenv()
os.makedirs("./logs", exist_ok=True)
bt.logging.set_trace()
bt.logging(debug=True, trace=False, logging_dir="./logs", record_log=True)


class Validator(BaseValidatorNeuron):
    """Poker44 validator neuron wired into the BaseValidator scaffold."""

    def __init__(self):
        cfg = config(Validator)
        super().__init__(config=cfg)
        bt.logging.info(f"🚀 Poker44 Validator v{__version__} started")

        self.forward_count = 0
        self.settings = cfg

        human_json_env = os.getenv("POKER44_HUMAN_JSON_PATH")
        if not human_json_env:
            raise RuntimeError(
                "POKER44_HUMAN_JSON_PATH must point to the private local human-hand JSON used by validators."
            )

        human_json_path = Path(human_json_env).expanduser().resolve()
        mixed_output_path = Path(
            os.getenv("POKER44_MIXED_DATASET_PATH", str(DEFAULT_OUTPUT_PATH))
        ).expanduser().resolve()
        refresh_seconds = int(
            os.getenv("POKER44_DATASET_REFRESH_SECONDS", str(60 * 60))
        )
        chunk_count = int(os.getenv("POKER44_CHUNK_COUNT", "40"))
        min_hands_per_chunk = int(os.getenv("POKER44_MIN_HANDS_PER_CHUNK", "60"))
        max_hands_per_chunk = int(os.getenv("POKER44_MAX_HANDS_PER_CHUNK", "120"))
        human_ratio = float(os.getenv("POKER44_HUMAN_RATIO", "0.5"))
        dataset_seed_env = os.getenv("POKER44_DATASET_SEED")
        dataset_seed = int(dataset_seed_env) if dataset_seed_env is not None else None
        self.chunk_batch_size = chunk_count
        self.dataset_cfg = MixedDatasetConfig(
            human_json_path=human_json_path,
            output_path=mixed_output_path,
            chunk_count=chunk_count,
            min_hands_per_chunk=min_hands_per_chunk,
            max_hands_per_chunk=max_hands_per_chunk,
            human_ratio=human_ratio,
            refresh_seconds=refresh_seconds,
            seed=dataset_seed,
        )
        self.provider = TimedMixedDatasetProvider(self.dataset_cfg)
        bt.logging.info(
            f"📁 Using mixed dataset provider | human_json={human_json_path} output={mixed_output_path} "
            f"chunks={chunk_count} hands_range=[{min_hands_per_chunk},{max_hands_per_chunk}] "
            f"ratio={human_ratio} refresh_s={refresh_seconds}"
        )
        bt.logging.info("🧭 Dataset generation is deterministic per refresh window.")
        configured_poll_interval = getattr(cfg, "poll_interval_seconds", refresh_seconds)
        self.poll_interval = int(
            os.getenv("POKER44_POLL_INTERVAL_SECONDS", str(configured_poll_interval))
        )
        self.reward_window = int(os.getenv("POKER44_REWARD_WINDOW", "40"))
        self.prediction_buffer = {}
        self.label_buffer = {}
        state_dir = Path(self.config.neuron.full_path)
        self.model_manifest_path = state_dir / "model_manifests.json"
        self.compliance_registry_path = state_dir / "compliance_registry.json"
        self.suspicion_registry_path = state_dir / "suspicion_registry.json"
        self.served_chunk_registry_path = state_dir / "served_chunk_registry.json"
        self.model_manifest_registry = load_json_registry(self.model_manifest_path)
        if self.model_manifest_registry:
            self.model_manifest_registry = normalize_uid_key_registry(
                self.model_manifest_registry
            )
        self.compliance_registry = load_json_registry(
            self.compliance_registry_path,
            default={"miners": {}, "summary": {}},
        )
        self.suspicion_registry = load_json_registry(
            self.suspicion_registry_path,
            default={"miners": {}, "summary": {}},
        )
        self.served_chunk_registry = load_json_registry(
            self.served_chunk_registry_path,
            default={"chunk_index": {}, "recent_cycles": [], "summary": {}},
        )
        self.wandb_helper = ValidatorWandbHelper(
            config=cfg,
            validator_uid=self.resolve_uid(self.wallet.hotkey.ss58_address),
            hotkey=self.wallet.hotkey.ss58_address,
            version=__version__,
            netuid=cfg.netuid,
        )
        self.wandb_helper.log_validator_startup(
            dataset_cfg=self.dataset_cfg,
            poll_interval=self.poll_interval,
            reward_window=self.reward_window,
        )

        self.forced_miner_uids: Optional[List[int]] = None
        self.forced_sleep_interval: Optional[float] = None
        self.forced_only_mode: bool = False

        force_uid_mode = os.getenv("POKER44_FORCE_UID6_MODE", "").strip().lower()
        self.forced_only_mode = force_uid_mode in {"1", "true", "yes", "uid6", "default"}

        force_uids_env = os.getenv("POKER44_FORCE_MINER_UIDS", "").strip()
        if self.forced_only_mode and not force_uids_env:
            force_uids_env = "6"

        if force_uids_env:
            parsed_force_uids: List[int] = []
            try:
                parsed_force_uids = [
                    int(uid.strip())
                    for uid in force_uids_env.split(",")
                    if uid.strip()
                ]
            except ValueError:
                bt.logging.warning(
                    "Invalid POKER44_FORCE_MINER_UIDS=%r; forcing disabled.",
                    force_uids_env,
                )
                parsed_force_uids = []

            if parsed_force_uids:
                self.forced_miner_uids = parsed_force_uids
                if self.forced_only_mode:
                    forced_sleep_env = os.getenv("POKER44_FORCE_POLL_SECONDS", "").strip()
                    if forced_sleep_env:
                        try:
                            self.forced_sleep_interval = max(0.2, float(forced_sleep_env))
                        except ValueError:
                            bt.logging.warning(
                                "Invalid POKER44_FORCE_POLL_SECONDS=%r; ignoring override.",
                                forced_sleep_env,
                            )
                            self.forced_sleep_interval = None
                    else:
                        self.forced_sleep_interval = None

                    bt.logging.warning(
                        "⚠️ Validator forcing miner UID(s) %s (override via POKER44_FORCE_MINER_UIDS).",
                        parsed_force_uids,
                    )
                    if self.forced_sleep_interval is not None:
                        bt.logging.warning(
                            "⚠️ Validator poll interval forced to %.2fs (POKER44_FORCE_POLL_SECONDS).",
                            self.forced_sleep_interval,
                        )
                else:
                    bt.logging.info(
                        "🔧 Validator will include forced miner UID(s) whenever possible: %s",
                        parsed_force_uids,
                    )

    def resolve_uid(self, hotkey: str) -> Optional[int]:
        try:
            return self.metagraph.hotkeys.index(hotkey)
        except ValueError:
            return None

    async def forward(self, synapse=None):  # type: ignore[override]
        return await forward_cycle(self)

    def __del__(self) -> None:
        wandb_helper = getattr(self, "wandb_helper", None)
        if wandb_helper is not None:
            try:
                wandb_helper.finish()
            except Exception:
                pass


if __name__ == "__main__":  # pragma: no cover - manual execution
    validator = Validator()
    validator.run()
