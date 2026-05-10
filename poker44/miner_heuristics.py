"""Standalone heuristic scoring for Poker44 miner (legacy + modern payloads)."""

from __future__ import annotations

import math
import os
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set
import pickle
from pathlib import Path
import time
import numpy as np

ZERO_STACK_EPS = 1e-3
STACK_THRESHOLD_BB_FRACTION = 0.05
UNIQUE_PLAYER_HUMAN_THRESHOLD = 60.0  # modern chunks have hundreds of unique IDs

# Decision-boundary calibration for single-hand chunks.
# The f0 model saturates (outputs ~1.0 for both classes) on 1-hand input, so no threshold
# value meaningfully helps until a dedicated single-hand model is trained.
# Dedicated single-hand v2 model is now integrated; keep natural 0.5 boundary.
# Formula: calibrated = clamp01(raw - SINGLE_HAND_BOT_THRESHOLD + 0.5)
# Default decision boundary for single-hand chunks.
# Some model families are calibrated with a lower boundary and are routed
# dynamically in _single_hand_bot_threshold().
SINGLE_HAND_BOT_THRESHOLD: float = 0.5


def _resolve_multihand_single_hand_ml_max_hands() -> int:
    """Resolve upper hand-count bound for short multihand ML vote routing."""
    raw = os.getenv("ML_MAX_HANDS", "40").strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 40
    return max(1, value)


# Route short multihand chunks through single-hand ML voting.
MULTIHAND_SINGLE_HAND_ML_MAX_HANDS: int = _resolve_multihand_single_hand_ml_max_hands()
REPO_ROOT = Path(__file__).resolve().parent.parent
ML_MODEL_PATH = REPO_ROOT / "weights" / "ml_filtered0_model.pkl"
ML_SCALER_PATH = REPO_ROOT / "weights" / "ml_filtered0_scaler.pkl"
ML_F1_MODEL_PATH = REPO_ROOT / "weights" / "ml_filtered1_nonhardcut_model.pkl"
ML_F1_SCALER_PATH = REPO_ROOT / "weights" / "ml_filtered1_nonhardcut_scaler.pkl"
ML_SINGLE_HAND_MODEL_PATH = REPO_ROOT / "weights" / "ml_single_hand_model.pkl"
ML_SINGLE_HAND_SCALER_PATH = REPO_ROOT / "weights" / "ml_single_hand_scaler.pkl"

SINGLE_HAND_MODEL_ALIASES: Dict[str, Tuple[str, str]] = {
    # Active runtime slot (default behavior).
    "active": ("weights/ml_single_hand_model.pkl", "weights/ml_single_hand_scaler.pkl"),
    # Historical / candidate aliases.
    "gen3": ("weights/ml_single_hand_v3_search_model.pkl", "weights/ml_single_hand_v3_search_scaler.pkl"),
    "gen4": ("weights/ml_single_hand_model.pkl", "weights/ml_single_hand_scaler.pkl"),
    "gen4_17": ("weights/ml_gen4_17_model.pkl", "weights/ml_gen4_17_scaler.pkl"),
    "gen4plus": ("weights/ml_single_hand_v4plus_s12346_model.pkl", "weights/ml_single_hand_v4plus_s12346_scaler.pkl"),
}


def configure_single_hand_model_paths(
    *,
    alias: Optional[str] = None,
    model_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
) -> None:
    """Configure single-hand model/scaler artifact paths before preload.

    Priority:
    1) explicit model_path/scaler_path
    2) alias from SINGLE_HAND_MODEL_ALIASES
    3) keep existing defaults
    """
    global ML_SINGLE_HAND_MODEL_PATH, ML_SINGLE_HAND_SCALER_PATH

    if alias:
        key = alias.strip().lower()
        if key in SINGLE_HAND_MODEL_ALIASES:
            rel_model, rel_scaler = SINGLE_HAND_MODEL_ALIASES[key]
            ML_SINGLE_HAND_MODEL_PATH = (REPO_ROOT / rel_model).resolve()
            ML_SINGLE_HAND_SCALER_PATH = (REPO_ROOT / rel_scaler).resolve()

    # Explicit paths have highest priority.
    if model_path is not None:
        ML_SINGLE_HAND_MODEL_PATH = Path(model_path).expanduser().resolve()
    if scaler_path is not None:
        ML_SINGLE_HAND_SCALER_PATH = Path(scaler_path).expanduser().resolve()


def configure_single_hand_model_paths_from_env() -> None:
    """Configure single-hand model/scaler paths from environment variables.

    Supported env vars:
    - POKER44_SINGLE_HAND_MODEL_ALIAS: one of active|gen3|gen4|gen4_17|gen4plus
    - POKER44_SINGLE_HAND_MODEL_PATH: absolute/relative path to model .pkl
    - POKER44_SINGLE_HAND_SCALER_PATH: absolute/relative path to scaler .pkl
    """
    alias = os.getenv("POKER44_SINGLE_HAND_MODEL_ALIAS", "").strip() or None
    model_env = os.getenv("POKER44_SINGLE_HAND_MODEL_PATH", "").strip() or None
    scaler_env = os.getenv("POKER44_SINGLE_HAND_SCALER_PATH", "").strip() or None

    model_path = Path(model_env) if model_env else None
    scaler_path = Path(scaler_env) if scaler_env else None
    configure_single_hand_model_paths(alias=alias, model_path=model_path, scaler_path=scaler_path)


def filter_other_actions_from_chunk(chunk: List[dict]) -> Tuple[List[dict], int]:
    """Remove action_type='other' entries from chunk payloads.

    Supports both payload shapes seen in miner inputs:
    - modern: chunk is a list of hands, each hand has ``actions`` list
    - flat/legacy-like: chunk is a list of action dicts

    Returns:
        (filtered_chunk, count_removed)
    """
    if not chunk:
        return chunk, 0

    removed_count = 0
    filtered_chunk: List[dict] = []

    for item in chunk:
        # Modern payload: a hand dict with nested actions.
        if isinstance(item, dict) and isinstance(item.get("actions"), list):
            actions = item.get("actions") or []
            filtered_actions = []
            local_removed = 0
            for action in actions:
                if isinstance(action, dict) and str(action.get("action_type") or "").strip().lower() == "other":
                    local_removed += 1
                    continue
                filtered_actions.append(action)

            if local_removed:
                new_item = dict(item)
                new_item["actions"] = filtered_actions
                filtered_chunk.append(new_item)
            else:
                filtered_chunk.append(item)

            removed_count += local_removed
            continue

        # Flat payload fallback: action records at chunk top level.
        if isinstance(item, dict) and str(item.get("action_type") or "").strip().lower() == "other":
            removed_count += 1
            continue

        filtered_chunk.append(item)

    return filtered_chunk, removed_count


def _load_ml_model_filtered0():
    """Lazy-load pre-trained ML model/scaler for filtered=0 scoring."""
    global _ML_FILTERED0_MODEL, _ML_FILTERED0_SCALER, _ML_MODEL_AVAILABLE
    global _ML_MODEL_LOAD_ATTEMPTS, _ML_MODEL_LOAD_SUCCESSES
    global _ML_MODEL_LAST_LOAD_TS, _ML_MODEL_LAST_ERROR, _ML_MODEL_LAST_SOURCE
    
    if _ML_MODEL_AVAILABLE or _ML_FILTERED0_MODEL is not None:
        return
    
    _ML_MODEL_LOAD_ATTEMPTS += 1
    try:
        if not ML_MODEL_PATH.exists() or not ML_SCALER_PATH.exists():
            _ML_MODEL_LAST_SOURCE = "missing_artifacts"
            _ML_MODEL_LAST_ERROR = "missing model/scaler artifacts in weights/"
            return

        with open(ML_MODEL_PATH, "rb") as f:
            _ML_FILTERED0_MODEL = pickle.load(f)
        with open(ML_SCALER_PATH, "rb") as f:
            _ML_FILTERED0_SCALER = pickle.load(f)

        _ML_MODEL_AVAILABLE = True
        _ML_MODEL_LOAD_SUCCESSES += 1
        _ML_MODEL_LAST_LOAD_TS = time.time()
        _ML_MODEL_LAST_SOURCE = "weights_artifacts"
        _ML_MODEL_LAST_ERROR = None
        
    except Exception:
        # Silent fail - will use heuristics as fallback
        _ML_FILTERED0_MODEL = None
        _ML_FILTERED0_SCALER = None
        _ML_MODEL_AVAILABLE = False
        _ML_MODEL_LAST_SOURCE = "weights_artifacts"
        _ML_MODEL_LAST_ERROR = "exception during artifact load"

def _load_ml_model_filtered1():
    """Lazy-load pre-trained ML model/scaler for filtered=1 non-hardcut scoring."""
    global _ML_FILTERED1_MODEL, _ML_FILTERED1_SCALER, _ML_F1_MODEL_AVAILABLE
    global _ML_F1_MODEL_LOAD_ATTEMPTS, _ML_F1_MODEL_LOAD_SUCCESSES
    global _ML_F1_MODEL_LAST_LOAD_TS, _ML_F1_MODEL_LAST_ERROR, _ML_F1_MODEL_LAST_SOURCE

    if _ML_F1_MODEL_AVAILABLE or _ML_FILTERED1_MODEL is not None:
        return

    _ML_F1_MODEL_LOAD_ATTEMPTS += 1
    try:
        if not ML_F1_MODEL_PATH.exists() or not ML_F1_SCALER_PATH.exists():
            _ML_F1_MODEL_LAST_SOURCE = "missing_artifacts"
            _ML_F1_MODEL_LAST_ERROR = "missing filtered1 model/scaler artifacts in weights/"
            return

        with open(ML_F1_MODEL_PATH, "rb") as f:
            _ML_FILTERED1_MODEL = pickle.load(f)
        with open(ML_F1_SCALER_PATH, "rb") as f:
            _ML_FILTERED1_SCALER = pickle.load(f)

        _ML_F1_MODEL_AVAILABLE = True
        _ML_F1_MODEL_LOAD_SUCCESSES += 1
        _ML_F1_MODEL_LAST_LOAD_TS = time.time()
        _ML_F1_MODEL_LAST_SOURCE = "weights_artifacts"
        _ML_F1_MODEL_LAST_ERROR = None

    except Exception:
        # Silent fail - will fall back to heuristic weights
        _ML_FILTERED1_MODEL = None
        _ML_FILTERED1_SCALER = None
        _ML_F1_MODEL_AVAILABLE = False
        _ML_F1_MODEL_LAST_SOURCE = "weights_artifacts"
        _ML_F1_MODEL_LAST_ERROR = "exception during filtered1 artifact load"


def _load_ml_model_single_hand():
    """Lazy-load pre-trained ML model/scaler for single-hand scoring."""
    global _ML_SINGLE_HAND_MODEL, _ML_SINGLE_HAND_SCALER, _ML_SINGLE_HAND_MODEL_AVAILABLE
    global _ML_SINGLE_HAND_LOAD_ATTEMPTS, _ML_SINGLE_HAND_LOAD_SUCCESSES
    global _ML_SINGLE_HAND_LAST_LOAD_TS, _ML_SINGLE_HAND_LAST_ERROR, _ML_SINGLE_HAND_LAST_SOURCE

    if _ML_SINGLE_HAND_MODEL_AVAILABLE or _ML_SINGLE_HAND_MODEL is not None:
        return

    _ML_SINGLE_HAND_LOAD_ATTEMPTS += 1
    try:
        if not ML_SINGLE_HAND_MODEL_PATH.exists():
            _ML_SINGLE_HAND_LAST_SOURCE = "missing_artifacts"
            _ML_SINGLE_HAND_LAST_ERROR = "missing single-hand model artifact in weights/"
            return

        with open(ML_SINGLE_HAND_MODEL_PATH, "rb") as f:
            _ML_SINGLE_HAND_MODEL = pickle.load(f)

        if ML_SINGLE_HAND_SCALER_PATH.exists():
            with open(ML_SINGLE_HAND_SCALER_PATH, "rb") as f:
                _ML_SINGLE_HAND_SCALER = pickle.load(f)
        else:
            _ML_SINGLE_HAND_SCALER = None

        _ML_SINGLE_HAND_MODEL_AVAILABLE = True
        _ML_SINGLE_HAND_LOAD_SUCCESSES += 1
        _ML_SINGLE_HAND_LAST_LOAD_TS = time.time()
        _ML_SINGLE_HAND_LAST_SOURCE = "weights_artifacts"
        _ML_SINGLE_HAND_LAST_ERROR = None

    except Exception:
        _ML_SINGLE_HAND_MODEL = None
        _ML_SINGLE_HAND_SCALER = None
        _ML_SINGLE_HAND_MODEL_AVAILABLE = False
        _ML_SINGLE_HAND_LAST_SOURCE = "weights_artifacts"
        _ML_SINGLE_HAND_LAST_ERROR = "exception during single-hand artifact load"

LEGACY_SB = 0.01
LEGACY_BB = 0.02
LEGACY_PLAYER_PREFIX = "seat_"
LEGACY_ACTION_PLACEHOLDERS = {"", "action"}

# ML model for filtered=0 scoring (lazy-loaded)
_ML_FILTERED0_MODEL = None
_ML_FILTERED0_SCALER = None
_ML_MODEL_AVAILABLE = False
_ML_MODEL_LOAD_ATTEMPTS = 0
_ML_MODEL_LOAD_SUCCESSES = 0
_ML_MODEL_LAST_LOAD_TS = None
_ML_MODEL_LAST_ERROR = None
_ML_MODEL_LAST_SOURCE = None
_ML_REQUEST_F0_ML_USED = 0
_ML_REQUEST_F0_HEUR_FALLBACK = 0

# ML model for filtered=1 non-hardcut scoring (lazy-loaded)
_ML_FILTERED1_MODEL = None
_ML_FILTERED1_SCALER = None
_ML_F1_MODEL_AVAILABLE = False
_ML_F1_MODEL_LOAD_ATTEMPTS = 0
_ML_F1_MODEL_LOAD_SUCCESSES = 0
_ML_F1_MODEL_LAST_LOAD_TS = None
_ML_F1_MODEL_LAST_ERROR = None
_ML_F1_MODEL_LAST_SOURCE = None
_ML_REQUEST_F1_ML_USED = 0
_ML_REQUEST_F1_HEUR_FALLBACK = 0
_ML_REQUEST_F1_HARDCUT_FORCED_HUMAN = 0
_ML_REQUEST_F2PLUS_FORCED_HUMAN = 0

# ML model for single-hand scoring (lazy-loaded)
_ML_SINGLE_HAND_MODEL = None
_ML_SINGLE_HAND_SCALER = None
_ML_SINGLE_HAND_MODEL_AVAILABLE = False
_ML_SINGLE_HAND_LOAD_ATTEMPTS = 0
_ML_SINGLE_HAND_LOAD_SUCCESSES = 0
_ML_SINGLE_HAND_LAST_LOAD_TS = None
_ML_SINGLE_HAND_LAST_ERROR = None
_ML_SINGLE_HAND_LAST_SOURCE = None
_ML_REQUEST_SINGLE_HAND_ML_USED = 0
_ML_REQUEST_SINGLE_HAND_F0_FALLBACK = 0
_ML_REQUEST_MULTIHAND_SMALL_ML_USED = 0
_ML_REQUEST_MULTIHAND_SMALL_HEUR_FALLBACK = 0


def reset_ml_request_stats() -> None:
    """Reset per-request ML usage counters."""
    global _ML_REQUEST_F0_ML_USED, _ML_REQUEST_F0_HEUR_FALLBACK
    global _ML_REQUEST_F1_ML_USED, _ML_REQUEST_F1_HEUR_FALLBACK
    global _ML_REQUEST_F1_HARDCUT_FORCED_HUMAN, _ML_REQUEST_F2PLUS_FORCED_HUMAN
    global _ML_REQUEST_SINGLE_HAND_ML_USED, _ML_REQUEST_SINGLE_HAND_F0_FALLBACK
    global _ML_REQUEST_MULTIHAND_SMALL_ML_USED, _ML_REQUEST_MULTIHAND_SMALL_HEUR_FALLBACK
    _ML_REQUEST_F0_ML_USED = 0
    _ML_REQUEST_F0_HEUR_FALLBACK = 0
    _ML_REQUEST_F1_ML_USED = 0
    _ML_REQUEST_F1_HEUR_FALLBACK = 0
    _ML_REQUEST_F1_HARDCUT_FORCED_HUMAN = 0
    _ML_REQUEST_F2PLUS_FORCED_HUMAN = 0
    _ML_REQUEST_SINGLE_HAND_ML_USED = 0
    _ML_REQUEST_SINGLE_HAND_F0_FALLBACK = 0
    _ML_REQUEST_MULTIHAND_SMALL_ML_USED = 0
    _ML_REQUEST_MULTIHAND_SMALL_HEUR_FALLBACK = 0


def get_ml_runtime_stats() -> Dict[str, object]:
    """Expose ML runtime diagnostics for miner logs and debugging."""
    return {
        "ml_model_available": _ML_MODEL_AVAILABLE,
        "ml_model_loaded": _ML_FILTERED0_MODEL is not None and _ML_FILTERED0_SCALER is not None,
        "ml_model_path": str(ML_MODEL_PATH),
        "ml_scaler_path": str(ML_SCALER_PATH),
        "ml_model_path_exists": ML_MODEL_PATH.exists(),
        "ml_scaler_path_exists": ML_SCALER_PATH.exists(),
        "ml_load_attempts": _ML_MODEL_LOAD_ATTEMPTS,
        "ml_load_successes": _ML_MODEL_LOAD_SUCCESSES,
        "ml_last_load_ts": _ML_MODEL_LAST_LOAD_TS,
        "ml_last_source": _ML_MODEL_LAST_SOURCE,
        "ml_last_error": _ML_MODEL_LAST_ERROR,
        "request_f0_ml_used": _ML_REQUEST_F0_ML_USED,
        "request_f0_heur_fallback": _ML_REQUEST_F0_HEUR_FALLBACK,
        "ml_f1_model_available": _ML_F1_MODEL_AVAILABLE,
        "ml_f1_model_loaded": _ML_FILTERED1_MODEL is not None and _ML_FILTERED1_SCALER is not None,
        "ml_f1_model_path": str(ML_F1_MODEL_PATH),
        "ml_f1_scaler_path": str(ML_F1_SCALER_PATH),
        "ml_f1_model_path_exists": ML_F1_MODEL_PATH.exists(),
        "ml_f1_scaler_path_exists": ML_F1_SCALER_PATH.exists(),
        "ml_f1_load_attempts": _ML_F1_MODEL_LOAD_ATTEMPTS,
        "ml_f1_load_successes": _ML_F1_MODEL_LOAD_SUCCESSES,
        "ml_f1_last_load_ts": _ML_F1_MODEL_LAST_LOAD_TS,
        "ml_f1_last_source": _ML_F1_MODEL_LAST_SOURCE,
        "ml_f1_last_error": _ML_F1_MODEL_LAST_ERROR,
        "request_f1_ml_used": _ML_REQUEST_F1_ML_USED,
        "request_f1_heur_fallback": _ML_REQUEST_F1_HEUR_FALLBACK,
        "request_f1_hardcut_forced_human": _ML_REQUEST_F1_HARDCUT_FORCED_HUMAN,
        "request_f2plus_forced_human": _ML_REQUEST_F2PLUS_FORCED_HUMAN,
        "ml_single_hand_model_available": _ML_SINGLE_HAND_MODEL_AVAILABLE,
        "ml_single_hand_model_loaded": _ML_SINGLE_HAND_MODEL is not None,
        "ml_single_hand_model_path": str(ML_SINGLE_HAND_MODEL_PATH),
        "ml_single_hand_scaler_path": str(ML_SINGLE_HAND_SCALER_PATH),
        "ml_single_hand_model_path_exists": ML_SINGLE_HAND_MODEL_PATH.exists(),
        "ml_single_hand_scaler_path_exists": ML_SINGLE_HAND_SCALER_PATH.exists(),
        "ml_single_hand_load_attempts": _ML_SINGLE_HAND_LOAD_ATTEMPTS,
        "ml_single_hand_load_successes": _ML_SINGLE_HAND_LOAD_SUCCESSES,
        "ml_single_hand_last_load_ts": _ML_SINGLE_HAND_LAST_LOAD_TS,
        "ml_single_hand_last_source": _ML_SINGLE_HAND_LAST_SOURCE,
        "ml_single_hand_last_error": _ML_SINGLE_HAND_LAST_ERROR,
        "request_single_hand_ml_used": _ML_REQUEST_SINGLE_HAND_ML_USED,
        "request_single_hand_f0_fallback": _ML_REQUEST_SINGLE_HAND_F0_FALLBACK,
        "request_multihand_small_ml_used": _ML_REQUEST_MULTIHAND_SMALL_ML_USED,
        "request_multihand_small_heur_fallback": _ML_REQUEST_MULTIHAND_SMALL_HEUR_FALLBACK,
    }

FILTERED0_WEIGHTS = {
    # Original metrics
    "multi_penalty": 0.7723998961364424,
    "multi_step": 0.16128092704799685,
    "street_floor": 0.7592138516029856,
    "street_span": 0.21472180013640746,
    "street_weight": 0.7429151690564527,
    "filled_threshold": 0.8281650685430967,
    "filled_boost": 0.20599045022733242,
    "filled_scale": 0.2849549001128677,
    "players_threshold": 4.795589889907139,
    "players_boost": 0.13656013580782722,
    "players_scale": 0.5281165652328261,
    # New metrics (action ratios, showdown, variance)
    "action_aggressiveness_weight": 0.1,  # raise/call ratio - bots are more passive
    "call_ratio_weight": 0.08,  # high call ratio suggests bot grinding
    "fold_ratio_weight": 0.05,  # low fold ratio suggests bot
    "showdown_freq_weight": 0.06,  # lower showdown freq suggests bot tight play
    "street_variance_weight": 0.04,  # variance in streets
    "player_volatility_weight": 0.03,  # variance in player count
}

FILTERED1_WEIGHTS = {
    "street_floor": 0.6761492895853173,
    "street_span": 0.19374406103339048,
    "street_weight": 0.5080490749398499,
    "filled_threshold": 0.9214846970169184,
    "filled_boost": 0.10920479423047504,
    "filled_scale": 0.36588399020446305,
    "players_threshold": 5.339764208229487,
    "players_boost": 0.10289470263337507,
    "players_scale": 0.4598083643285187,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _player_uids(hand: dict) -> Set[str]:
    return {p.get("player_uid") for p in hand.get("players", []) if p.get("player_uid")}


def _stack_threshold(hand: dict) -> float:
    metadata = hand.get("metadata") or {}
    bb = float(metadata.get("bb") or 0.0)
    if bb <= 0:
        bb = 0.05
    return max(ZERO_STACK_EPS, bb * STACK_THRESHOLD_BB_FRACTION)


def _compute_busted_players(hand: dict) -> Set[str]:
    players = hand.get("players") or []
    seat_to_uid = {p.get("seat"): p.get("player_uid") for p in players if p.get("seat")}
    start_stack = {
        p.get("player_uid"): float(p.get("starting_stack") or 0.0)
        for p in players
        if p.get("player_uid")
    }

    contributions = {uid: 0.0 for uid in start_stack.keys()}
    for action in hand.get("actions") or []:
        seat = action.get("actor_seat")
        uid = seat_to_uid.get(seat)
        if not uid:
            continue
        amount = float(action.get("amount") or 0.0)
        if amount == 0:
            continue
        if action.get("action_type") == "uncalled_bet_return":
            contributions[uid] -= amount
        else:
            contributions[uid] += amount

    payouts = (hand.get("outcome") or {}).get("payouts") or {}
    threshold = _stack_threshold(hand)
    busted: Set[str] = set()
    for uid, starting in start_stack.items():
        invested = contributions.get(uid, 0.0)
        payout = float(payouts.get(uid, 0.0))
        final_stack = starting - invested + payout
        if final_stack <= threshold:
            busted.add(uid)
    return busted


def _multi_leave_stats(chunk: List[dict]) -> Tuple[int, int, int, int]:
    filtered_multi_leave = 0
    raw_multi_leave = 0
    multi_joinleave = 0
    total_transitions = 0
    prev_players: Optional[Set[str]] = None
    prev_busted: Set[str] = set()

    for hand in chunk:
        players = _player_uids(hand)
        busted_players = _compute_busted_players(hand)

        if prev_players is not None:
            prev_only = prev_players - players
            joined = players - prev_players
            if len(prev_only) >= 2 or len(joined) >= 2:
                multi_joinleave += 1
            if len(prev_only) >= 2:
                raw_multi_leave += 1
                actual_leaves = prev_only - prev_busted
                if len(actual_leaves) >= 2:
                    filtered_multi_leave += 1
            total_transitions += 1

        prev_players = players
        prev_busted = busted_players

    return filtered_multi_leave, total_transitions, multi_joinleave, raw_multi_leave


def _score_hand_legacy(hand: dict) -> float:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    outcome = hand.get("outcome") or {}

    action_counts = Counter(action.get("action_type") for action in actions)
    meaningful_actions = max(
        1,
        sum(
            action_counts.get(kind, 0)
            for kind in ("call", "check", "bet", "raise", "fold")
        ),
    )

    call_ratio = action_counts.get("call", 0) / meaningful_actions
    check_ratio = action_counts.get("check", 0) / meaningful_actions
    fold_ratio = action_counts.get("fold", 0) / meaningful_actions
    raise_ratio = action_counts.get("raise", 0) / meaningful_actions
    street_depth = len(streets) / 3.0
    showdown_flag = 1.0 if outcome.get("showdown") else 0.0

    player_count_signal = 0.0
    if players:
        player_count_signal = (6 - min(len(players), 6)) / 4.0

    score = 0.0
    score += 0.32 * street_depth
    score += 0.22 * showdown_flag
    score += 0.18 * _clamp01(call_ratio / 0.35)
    score += 0.12 * _clamp01(check_ratio / 0.30)
    score += 0.08 * _clamp01(player_count_signal)
    score -= 0.18 * _clamp01(fold_ratio / 0.55)
    score -= 0.10 * _clamp01(raise_ratio / 0.20)

    return _clamp01(score)


def _unique_player_score(chunk: List[dict]) -> float:
    unique_players = set()
    for hand in chunk:
        unique_players.update(_player_uids(hand))
    count = len(unique_players)
    if count >= UNIQUE_PLAYER_HUMAN_THRESHOLD:
        return 0.0
    return _clamp01(1.0 - (count / UNIQUE_PLAYER_HUMAN_THRESHOLD))


def _avg_players_per_hand_chunk(chunk: List[dict]) -> float:
    total_players = 0
    hands = 0
    for hand in chunk:
        players = hand.get("players") or []
        total_players += len(players)
        hands += 1
    if hands == 0:
        return 0.0
    return total_players / hands


def _avg_filled_seat_ratio(chunk: List[dict]) -> float:
    total_ratio = 0.0
    hands = 0
    for hand in chunk:
        players = hand.get("players") or []
        metadata = hand.get("metadata") or {}
        max_seats = int(metadata.get("max_seats") or 0) or 6
        ratio = len(players) / max(1, max_seats)
        total_ratio += ratio
        hands += 1
    if hands == 0:
        return 0.0
    return total_ratio / hands


def _extract_actor_seat(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value).strip())
    except Exception:
        return 0


def _hand_has_modern_actions(hand: dict) -> bool:
    actions = hand.get("actions") or []
    for action in actions:
        action_type = str(action.get("action_type") or "").strip().lower()
        if action_type and action_type not in LEGACY_ACTION_PLACEHOLDERS:
            return True
        actor_seat = _extract_actor_seat(action.get("actor_seat"))
        if actor_seat > 0:
            return True
        normalized_amount = float(action.get("normalized_amount_bb") or 0.0)
        amount = float(action.get("amount") or 0.0)
        if normalized_amount != 0.0 or amount != 0.0:
            return True
    return False


def _avg_streets_per_hand(chunk: List[dict]) -> float:
    streets = 0
    hands = 0
    for hand in chunk:
        streets += len(hand.get("streets") or [])
        hands += 1
    if hands == 0:
        return 0.0
    return streets / (hands or 1)


def _action_ratios_chunk(chunk: List[dict]) -> Tuple[float, float, float, float]:
    """Compute average action ratios (call, check, fold, raise) across all hands in chunk."""
    total_call = 0
    total_check = 0
    total_fold = 0
    total_raise = 0
    total_meaningful = 0
    
    for hand in chunk:
        actions = hand.get("actions") or []
        action_counts = Counter(action.get("action_type") for action in actions)
        meaningful = max(1, sum(action_counts.get(kind, 0) for kind in ("call", "check", "bet", "raise", "fold")))
        
        total_call += action_counts.get("call", 0)
        total_check += action_counts.get("check", 0)
        total_fold += action_counts.get("fold", 0)
        total_raise += action_counts.get("raise", 0)
        total_meaningful += meaningful
    
    if total_meaningful == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    call_ratio = total_call / total_meaningful
    check_ratio = total_check / total_meaningful
    fold_ratio = total_fold / total_meaningful
    raise_ratio = total_raise / total_meaningful
    
    return call_ratio, check_ratio, fold_ratio, raise_ratio


def _showdown_frequency(chunk: List[dict]) -> float:
    """Compute proportion of hands that went to showdown."""
    showdown_count = 0
    total = 0
    
    for hand in chunk:
        outcome = hand.get("outcome") or {}
        total += 1
        if outcome.get("showdown"):
            showdown_count += 1
    
    if total == 0:
        return 0.0
    return showdown_count / total


def _street_variance(chunk: List[dict]) -> float:
    """Compute variance in number of streets across hands."""
    streets_per_hand = []
    for hand in chunk:
        streets = len(hand.get("streets") or [])
        streets_per_hand.append(streets)
    
    if len(streets_per_hand) < 2:
        return 0.0
    
    mean = sum(streets_per_hand) / len(streets_per_hand)
    variance = sum((x - mean) ** 2 for x in streets_per_hand) / len(streets_per_hand)
    return variance ** 0.5  # return std dev instead of variance


def _player_volatility(chunk: List[dict]) -> float:
    """Compute variance in number of players across hands."""
    player_counts = []
    for hand in chunk:
        players = hand.get("players") or []
        player_counts.append(len(players))
    
    if len(player_counts) < 2:
        return 0.0
    
    mean = sum(player_counts) / len(player_counts)
    variance = sum((x - mean) ** 2 for x in player_counts) / len(player_counts)
    return variance ** 0.5  # return std dev instead of variance


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _single_hand_bb(hand: dict) -> float:
    bb = _safe_float((hand.get("metadata") or {}).get("bb"))
    return bb if bb > 0 else 0.05


def _extract_ml_features_single_hand_v2(chunk: List[dict]) -> Optional[np.ndarray]:
    """Extract 25-dimensional hand-level features for single-hand v2 models."""
    if not chunk:
        return None
    hand = chunk[0]

    players = hand.get("players") or []
    actions = hand.get("actions") or []
    outcome = hand.get("outcome") or {}
    streets = hand.get("streets") or []
    metadata = hand.get("metadata") or {}

    bb = _single_hand_bb(hand)
    max_seats = int(metadata.get("max_seats") or 6)
    max_seats = max(max_seats, 1)

    num_players = float(len(players))
    filled_ratio = num_players / float(max_seats)

    starting_stacks = [_safe_float(p.get("starting_stack")) for p in players]
    stack_mean = float(np.mean(starting_stacks)) if starting_stacks else 0.0
    stack_std = float(np.std(starting_stacks)) if starting_stacks else 0.0
    stack_cv = stack_std / (stack_mean + 1e-9)

    action_types = [str(a.get("action_type") or "").lower() for a in actions]
    total_actions = float(len(action_types))

    def _count(name: str) -> float:
        return float(sum(1 for t in action_types if t == name))

    call_c = _count("call")
    check_c = _count("check")
    fold_c = _count("fold")
    raise_c = _count("raise")
    bet_c = _count("bet")
    allin_c = float(sum(1 for t in action_types if "all_in" in t or "all-in" in t))

    meaningful = call_c + check_c + fold_c + raise_c + bet_c
    if meaningful > 0:
        call_r = call_c / meaningful
        check_r = check_c / meaningful
        fold_r = fold_c / meaningful
        raise_r = raise_c / meaningful
        bet_r = bet_c / meaningful
    else:
        call_r = check_r = fold_r = raise_r = bet_r = 0.0

    agg_ratio = (raise_c + bet_c) / (call_c + check_c + 1.0)

    amounts = [_safe_float(a.get("amount")) for a in actions]
    amounts_pos = [a for a in amounts if a > 0]
    amount_mean_bb = (float(np.mean(amounts_pos)) / bb) if amounts_pos else 0.0
    amount_max_bb = (float(np.max(amounts_pos)) / bb) if amounts_pos else 0.0
    amount_std_bb = (float(np.std(amounts_pos)) / bb) if len(amounts_pos) > 1 else 0.0

    total_pot = _safe_float(outcome.get("total_pot"))
    total_pot_bb = total_pot / bb
    showdown = 1.0 if bool(outcome.get("showdown")) else 0.0
    payouts = outcome.get("payouts") or {}
    winner_count = float(sum(1 for _, v in payouts.items() if _safe_float(v) > 0))
    winner_share = winner_count / (num_players + 1e-9)

    n_streets = float(len(streets))
    actions_per_player = total_actions / (num_players + 1e-9)
    actions_per_street = total_actions / (n_streets + 1e-9)

    features = np.array(
        [
            num_players,
            float(max_seats),
            filled_ratio,
            stack_mean,
            stack_std,
            stack_cv,
            total_actions,
            meaningful,
            call_r,
            check_r,
            fold_r,
            raise_r,
            bet_r,
            agg_ratio,
            allin_c,
            amount_mean_bb,
            amount_max_bb,
            amount_std_bb,
            total_pot_bb,
            showdown,
            winner_count,
            winner_share,
            n_streets,
            actions_per_player,
            actions_per_street,
        ],
        dtype=np.float32,
    )
    return features


def _extract_ml_features_gen4(chunk: List[dict]) -> Optional[np.ndarray]:
    """Extract 16-dimensional feature vector for Gen4 RandomForest model (v4_s12346).

    Feature order (must match train_gen4_model.extract_single_hand_features):
    0  num_players       1  filled_ratio     2  stack_mean       3  stack_std
    4  stack_cv          5  total_actions    6  call_r           7  check_r
    8  fold_r            9  raise_r          10 agg_ratio        11 amount_mean_bb
    12 amount_max_bb     13 total_pot_bb     14 showdown         15 street_depth
    """
    if not chunk:
        return None
    hand = chunk[0]

    players = hand.get("players") or []
    actions = hand.get("actions") or []
    outcome = hand.get("outcome") or {}
    streets = hand.get("streets") or []
    metadata = hand.get("metadata") or {}

    try:
        hand_bb_val = float(metadata.get("bb") or 0.01) or 0.01
    except (TypeError, ValueError):
        hand_bb_val = 0.01
    max_seats = max(int(metadata.get("max_seats") or 6), 1)

    num_players = float(len(players))
    filled_ratio = num_players / float(max_seats)

    starting_stacks = []
    for p in players:
        try:
            starting_stacks.append(float(p.get("starting_stack") or 0))
        except (TypeError, ValueError):
            starting_stacks.append(0.0)
    # Scale stacks to match training data distribution.
    # Training p90 stack_mean_bb ~140; live data p90 ~5. Factor = 140/5 = 28.
    # Training raw stack mean ~$2.36; live raw stack mean ~$0.16. Factor = 2.36/0.16 ≈ 15.
    _STACK_FACTOR = 15.0
    starting_stacks_scaled = [s * _STACK_FACTOR for s in starting_stacks]
    stack_mean = float(np.mean(starting_stacks_scaled)) if starting_stacks_scaled else 0.0
    stack_std = float(np.std(starting_stacks_scaled)) if starting_stacks_scaled else 0.0
    stack_cv = stack_std / (stack_mean + 1e-9)

    action_types = [str(a.get("action_type") or "").lower() for a in actions]
    total_actions = float(len(action_types))

    def _cnt(name: str) -> float:
        return float(sum(1 for t in action_types if t == name))

    call_c = _cnt("call")
    check_c = _cnt("check")
    fold_c = _cnt("fold")
    raise_c = _cnt("raise")
    bet_c = _cnt("bet")

    meaningful = call_c + check_c + fold_c + raise_c + bet_c
    if meaningful > 0:
        call_r = call_c / meaningful
        check_r = check_c / meaningful
        fold_r = fold_c / meaningful
        raise_r = raise_c / meaningful
    else:
        call_r = check_r = fold_r = raise_r = 0.0

    agg_ratio = (raise_c + bet_c) / (call_c + check_c + 1.0)

    amounts = []
    for a in actions:
        try:
            amounts.append(float(a.get("amount") or 0))
        except (TypeError, ValueError):
            amounts.append(0.0)
    amounts_pos = [a for a in amounts if a > 0]
    amount_mean_bb = (float(np.mean(amounts_pos)) / hand_bb_val) if amounts_pos else 0.0
    amount_max_bb = (float(np.max(amounts_pos)) / hand_bb_val) if amounts_pos else 0.0

    try:
        total_pot = float(outcome.get("total_pot") or 0)
    except (TypeError, ValueError):
        total_pot = 0.0
    total_pot_bb = total_pot / hand_bb_val
    showdown = 1.0 if bool(outcome.get("showdown")) else 0.0

    flop_seen = 1.0 if any(s.get("street") == "FLOP" for s in streets) else 0.0
    turn_seen = 1.0 if any(s.get("street") == "TURN" for s in streets) else 0.0
    river_seen = 1.0 if any(s.get("street") == "RIVER" for s in streets) else 0.0
    street_depth = flop_seen + turn_seen + river_seen

    return np.array(
        [
            num_players,
            filled_ratio,
            stack_mean,
            stack_std,
            stack_cv,
            total_actions,
            call_r,
            check_r,
            fold_r,
            raise_r,
            agg_ratio,
            amount_mean_bb,
            amount_max_bb,
            total_pot_bb,
            showdown,
            street_depth,
        ],
        dtype=np.float32,
    )


def _extract_ml_features_gen4_17(chunk: List[dict]) -> Optional[np.ndarray]:
    """Extract 17-dimensional feature vector for Gen4.17 model.

    Feature order (must match train_gen4_17_model.extract_single_hand_features):
    0  num_players       1  filled_ratio     2  stack_mean       3  stack_std
    4  stack_cv          5  total_actions    6  call_r           7  check_r
    8  fold_r            9  raise_r          10 agg_ratio        11 amount_mean_bb
    12 amount_max_bb     13 amount_std_bb    14 total_pot_bb     15 showdown
    16 street_depth
    """
    if not chunk:
        return None
    hand = chunk[0]

    players = hand.get("players") or []
    actions = hand.get("actions") or []
    outcome = hand.get("outcome") or {}
    streets = hand.get("streets") or []
    metadata = hand.get("metadata") or {}

    try:
        hand_bb_val = float(metadata.get("bb") or 0.01) or 0.01
    except (TypeError, ValueError):
        hand_bb_val = 0.01
    max_seats = max(int(metadata.get("max_seats") or 6), 1)

    num_players = float(len(players))
    filled_ratio = num_players / float(max_seats)

    starting_stacks = []
    for p in players:
        try:
            starting_stacks.append(float(p.get("starting_stack") or 0))
        except (TypeError, ValueError):
            starting_stacks.append(0.0)
    stack_mean = float(np.mean(starting_stacks)) if starting_stacks else 0.0
    stack_std = float(np.std(starting_stacks)) if starting_stacks else 0.0
    stack_cv = stack_std / (stack_mean + 1e-9)

    action_types = [str(a.get("action_type") or "").lower() for a in actions]
    total_actions = float(len(action_types))

    def _cnt(name: str) -> float:
        return float(sum(1 for t in action_types if t == name))

    call_c = _cnt("call")
    check_c = _cnt("check")
    fold_c = _cnt("fold")
    raise_c = _cnt("raise")
    bet_c = _cnt("bet")

    meaningful = call_c + check_c + fold_c + raise_c + bet_c
    if meaningful > 0:
        call_r = call_c / meaningful
        check_r = check_c / meaningful
        fold_r = fold_c / meaningful
        raise_r = raise_c / meaningful
    else:
        call_r = check_r = fold_r = raise_r = 0.0

    agg_ratio = (raise_c + bet_c) / (call_c + check_c + 1.0)

    amounts = []
    for a in actions:
        try:
            amounts.append(float(a.get("amount") or 0))
        except (TypeError, ValueError):
            amounts.append(0.0)
    amounts_pos = [a for a in amounts if a > 0]
    amount_mean_bb = (float(np.mean(amounts_pos)) / hand_bb_val) if amounts_pos else 0.0
    amount_max_bb = (float(np.max(amounts_pos)) / hand_bb_val) if amounts_pos else 0.0
    amount_std_bb = (float(np.std(amounts_pos)) / hand_bb_val) if len(amounts_pos) > 1 else 0.0

    try:
        total_pot = float(outcome.get("total_pot") or 0)
    except (TypeError, ValueError):
        total_pot = 0.0
    total_pot_bb = total_pot / hand_bb_val
    showdown = 1.0 if bool(outcome.get("showdown")) else 0.0

    flop_seen = 1.0 if any(s.get("street") == "FLOP" for s in streets) else 0.0
    turn_seen = 1.0 if any(s.get("street") == "TURN" for s in streets) else 0.0
    river_seen = 1.0 if any(s.get("street") == "RIVER" for s in streets) else 0.0
    street_depth = flop_seen + turn_seen + river_seen

    return np.array(
        [
            num_players,
            filled_ratio,
            stack_mean,
            stack_std,
            stack_cv,
            total_actions,
            call_r,
            check_r,
            fold_r,
            raise_r,
            agg_ratio,
            amount_mean_bb,
            amount_max_bb,
            amount_std_bb,
            total_pot_bb,
            showdown,
            street_depth,
        ],
        dtype=np.float32,
    )


def _extract_ml_features_filtered0(chunk: List[dict]) -> Optional[np.ndarray]:
    """Extract 16-dimensional feature vector for ML model on filtered=0 chunk."""
    if not chunk:
        return None
    
    features = []
    
    # Action ratios (5 features)
    call_count = check_count = fold_count = raise_count = bet_count = 0
    total_meaningful = 0
    
    for hand in chunk:
        actions = hand.get("actions") or []
        action_counts = Counter(action.get("action_type") for action in actions)
        meaningful = max(1, sum(action_counts.get(kind, 0) for kind in ("call", "check", "bet", "raise", "fold")))
        
        call_count += action_counts.get("call", 0)
        check_count += action_counts.get("check", 0)
        fold_count += action_counts.get("fold", 0)
        raise_count += action_counts.get("raise", 0)
        bet_count += action_counts.get("bet", 0)
        total_meaningful += meaningful
    
    if total_meaningful > 0:
        features.extend([
            call_count / total_meaningful,
            check_count / total_meaningful,
            fold_count / total_meaningful,
            raise_count / total_meaningful,
            bet_count / total_meaningful,
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Street stats (4 features)
    streets_per_hand = [len(h.get("streets") or []) for h in chunk]
    if streets_per_hand:
        features.extend([
            np.mean(streets_per_hand),
            np.std(streets_per_hand) if len(streets_per_hand) > 1 else 0.0,
            float(np.min(streets_per_hand)),
            float(np.max(streets_per_hand)),
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
    
    # Player stats (3 features)
    player_counts = [len(h.get("players") or []) for h in chunk]
    if player_counts:
        features.extend([
            np.mean(player_counts),
            np.std(player_counts) if len(player_counts) > 1 else 0.0,
        ])
    else:
        features.extend([0.0, 0.0])
    
    unique_players = set()
    for hand in chunk:
        unique_players.update(p.get("player_uid") for p in hand.get("players", []) if p.get("player_uid"))
    features.append(float(len(unique_players)))
    
    # Showdown frequency (1 feature)
    showdown_count = sum(1 for h in chunk if (h.get("outcome") or {}).get("showdown"))
    features.append(showdown_count / len(chunk) if chunk else 0.0)
    
    # Filled seat ratio (1 feature)
    filled_ratios = []
    for hand in chunk:
        players = hand.get("players") or []
        max_seats = int((hand.get("metadata") or {}).get("max_seats") or 6)
        if max_seats > 0:
            filled_ratios.append(len(players) / max_seats)
    features.append(np.mean(filled_ratios) if filled_ratios else 0.0)
    
    # Pot stats (2 features)
    pots = [float((h.get("outcome") or {}).get("total_pot") or 0.0) for h in chunk]
    if pots:
        features.extend([
            np.mean(pots),
            np.std(pots) if len(pots) > 1 else 0.0,
        ])
    else:
        features.extend([0.0, 0.0])
    
    return np.array(features, dtype=np.float32)


def _score_filtered_one_ml(chunk: List[dict]) -> Optional[float]:
    """
    Score chunk using ML model for filtered=1 non-hardcut cases.
    Returns class-1 probability (bot score) or None if model unavailable.
    """
    if not _ML_F1_MODEL_AVAILABLE or _ML_FILTERED1_MODEL is None or _ML_FILTERED1_SCALER is None:
        return None

    try:
        features = _extract_ml_features_filtered0(chunk)
        if features is None:
            return None

        features_scaled = _ML_FILTERED1_SCALER.transform(features.reshape(1, -1))
        class1_prob = _ML_FILTERED1_MODEL.predict_proba(features_scaled)[0, 1]
        return float(class1_prob)

    except Exception:
        return None


def _score_filtered_zero_ml(chunk: List[dict]) -> Optional[float]:
    """
    Score chunk using ML model for filtered=0.
    Returns class-1 probability from the trained classifier.
    In this project class-1 is interpreted downstream as bot-risk
    Score >= 0.5 means bot prediction.
    Returns None if ML model unavailable.
    """
    if not _ML_MODEL_AVAILABLE or _ML_FILTERED0_MODEL is None or _ML_FILTERED0_SCALER is None:
        return None
    
    try:
        features = _extract_ml_features_filtered0(chunk)
        if features is None:
            return None
        
        features_scaled = _ML_FILTERED0_SCALER.transform(features.reshape(1, -1))
        
        # Use class-1 probability directly for downstream bot thresholding.
        class1_prob = _ML_FILTERED0_MODEL.predict_proba(features_scaled)[0, 1]

        return float(class1_prob)
        
    except Exception:
        # Silent fail - will fallback to heuristics
        return None


def _score_single_hand_ml(chunk: List[dict]) -> Optional[float]:
    """Score single-hand chunk using dedicated model if available."""
    global _ML_SINGLE_HAND_LAST_ERROR

    if not _ML_SINGLE_HAND_MODEL_AVAILABLE or _ML_SINGLE_HAND_MODEL is None:
        return None

    try:
        extractor_fn = _single_hand_feature_extractor_for_model()
        features = extractor_fn(chunk)

        if features is None:
            return None

        features_batch = features.reshape(1, -1)
        if _ML_SINGLE_HAND_SCALER is not None:
            features_batch = _ML_SINGLE_HAND_SCALER.transform(features_batch)

        class1_prob = _ML_SINGLE_HAND_MODEL.predict_proba(features_batch)[0, 1]
        _ML_SINGLE_HAND_LAST_ERROR = None
        return float(class1_prob)

    except Exception as exc:
        _ML_SINGLE_HAND_LAST_ERROR = f"single_hand_scoring_error:{type(exc).__name__}"
        return None


def _single_hand_feature_extractor_for_model():
    """Select feature extractor for currently loaded single-hand model."""
    expected_features = getattr(_ML_SINGLE_HAND_MODEL, "n_features_in_", None)
    scaler_expected_features = getattr(_ML_SINGLE_HAND_SCALER, "n_features_in_", None)
    extractor_tag = getattr(_ML_SINGLE_HAND_MODEL, "_feature_extractor_tag", None)

    # Prefer model metadata; fallback to scaler metadata if missing.
    feature_count = expected_features if expected_features is not None else scaler_expected_features

    # Route by explicit model tag first, then by feature dimensionality.
    if extractor_tag == "gen4_17_v1" or feature_count == 17:
        return _extract_ml_features_gen4_17
    if extractor_tag == "gen4_v1":
        return _extract_ml_features_gen4
    if feature_count == 25:
        return _extract_ml_features_single_hand_v2
    if feature_count == 16 and extractor_tag is None:
        # Older single-hand 16-feature models were trained with the Gen4 extractor.
        return _extract_ml_features_gen4
    # Safe fallback for unknown model variants.
    return _extract_ml_features_filtered0


def _score_small_multihand_single_hand_vote(chunk: List[dict]) -> Optional[float]:
    """Score short multihand chunks by single-hand ML voting (2..ML_MAX_HANDS hands)."""
    global _ML_REQUEST_MULTIHAND_SMALL_ML_USED

    _load_ml_model_single_hand()
    if _ML_SINGLE_HAND_MODEL is None:
        return None

    try:
        extractor_fn = _single_hand_feature_extractor_for_model()
        feature_rows = []
        for hand in chunk:
            features = extractor_fn([hand])
            if features is None:
                continue
            feature_rows.append(features)

        if not feature_rows:
            return None

        features_batch = np.asarray(feature_rows, dtype=np.float32)
        if _ML_SINGLE_HAND_SCALER is not None:
            features_batch = _ML_SINGLE_HAND_SCALER.transform(features_batch)

        raw_probs = _ML_SINGLE_HAND_MODEL.predict_proba(features_batch)[:, 1]
        single_hand_threshold = _single_hand_bot_threshold()
        calibrated = np.clip(raw_probs - single_hand_threshold + 0.5, 0.0, 1.0)
        bot_flags = calibrated >= 0.5
        if bot_flags.size == 0:
            return None

        _ML_REQUEST_MULTIHAND_SMALL_ML_USED += 1
        return round(float(np.mean(bot_flags.astype(np.float32))), 6)
    except Exception as exc:
        global _ML_SINGLE_HAND_LAST_ERROR
        _ML_SINGLE_HAND_LAST_ERROR = f"small_multihand_single_hand_scoring_error:{type(exc).__name__}"
        return None


def _single_hand_bot_threshold() -> float:
    """Return calibrated single-hand threshold for the loaded model family."""
    model = _ML_SINGLE_HAND_MODEL
    if model is None:
        return SINGLE_HAND_BOT_THRESHOLD

    extractor_tag = getattr(model, "_feature_extractor_tag", None)
    if extractor_tag in {"gen4_v1", "gen4_17_v1", "gen5_v1", "gen5_17_v1"}:
        return 0.3

    # Some historical Gen5 artifacts are untagged. Identify them by model metadata
    # and active artifact file name so calibration remains deterministic.
    model_name = str(getattr(model, "_model_name", "") or "").lower()
    if "gen5" in model_name or "v5" in model_name:
        return 0.3

    try:
        active_model_name = ML_SINGLE_HAND_MODEL_PATH.resolve().name.lower()
    except Exception:
        active_model_name = ML_SINGLE_HAND_MODEL_PATH.name.lower()
    if "gen5" in active_model_name or "v5" in active_model_name:
        return 0.3

    return SINGLE_HAND_BOT_THRESHOLD


def _score_single_hand_chunk_with_route(chunk: List[dict]) -> Tuple[float, str]:
    """
    Dedicated scoring path for single-hand chunks.

    Uses dedicated single-hand model if available. Falls back to the filtered=0
    path otherwise.
    """
    global _ML_REQUEST_SINGLE_HAND_ML_USED, _ML_REQUEST_SINGLE_HAND_F0_FALLBACK

    _load_ml_model_single_hand()
    single_hand_threshold = _single_hand_bot_threshold()
    if _ML_SINGLE_HAND_MODEL is not None:
        ml_score = _score_single_hand_ml(chunk)
        if ml_score is not None:
            _ML_REQUEST_SINGLE_HAND_ML_USED += 1
            raw = _clamp01(ml_score)
            return _clamp01(raw - single_hand_threshold + 0.5), "1-handed_model"

    _ML_REQUEST_SINGLE_HAND_F0_FALLBACK += 1
    raw = _score_filtered_zero(chunk, multi_joinleave=0, raw_multi_leave=0)
    return _clamp01(raw - single_hand_threshold + 0.5), "1-handed_model"


def _score_single_hand_chunk(chunk: List[dict]) -> float:
    score, _ = _score_single_hand_chunk_with_route(chunk)
    return score


def score_chunk_modern_with_route(chunk: List[dict]) -> Tuple[float, str]:
    global _ML_REQUEST_F2PLUS_FORCED_HUMAN, _ML_REQUEST_MULTIHAND_SMALL_HEUR_FALLBACK

    if not chunk:
        return 0.5, "empty_chunk"

    # Single-hand provider-runtime style chunks use a dedicated route before
    # filtered bucket checks. The model implementation can be swapped later.
    if len(chunk) == 1:
        return _score_single_hand_chunk_with_route(chunk)

    # Short multihand chunks (2..40) are scored by per-hand ML voting.
    if 2 <= len(chunk) <= MULTIHAND_SINGLE_HAND_ML_MAX_HANDS:
        ml_vote_score = _score_small_multihand_single_hand_vote(chunk)
        if ml_vote_score is not None:
            return ml_vote_score, "small_multihand_ml"
        _ML_REQUEST_MULTIHAND_SMALL_HEUR_FALLBACK += 1

    # Lazy-load legacy multihand ML/heuristic artifacts for >=41 hands and fallbacks.
    _load_ml_model_filtered0()
    _load_ml_model_filtered1()

    filtered_multi_leave, _, multi_joinleave, raw_multi_leave = _multi_leave_stats(chunk)

    if filtered_multi_leave >= 2:
        _ML_REQUEST_F2PLUS_FORCED_HUMAN += 1
        return 0.0, "multihand_heuristics"

    if filtered_multi_leave == 1:
        return _score_filtered_one(chunk, multi_joinleave, raw_multi_leave), "multihand_heuristics"

    # filtered_multi_leave == 0
    return _score_filtered_zero(chunk, multi_joinleave, raw_multi_leave), "multihand_heuristics"


def score_chunk_modern(chunk: List[dict]) -> float:
    score, _route = score_chunk_modern_with_route(chunk)
    return score


def score_chunk_legacy(chunk: List[dict]) -> float:
    if not chunk:
        return 0.5

    hand_scores = [_score_hand_legacy(hand) for hand in chunk]
    avg_score = sum(hand_scores) / len(hand_scores)
    filtered_multi_leave, _, _, _ = _multi_leave_stats(chunk)
    multi_leave_signal = math.exp(-0.85 * filtered_multi_leave)
    if filtered_multi_leave == 0:
        multi_leave_signal += 0.15
    elif filtered_multi_leave == 1:
        multi_leave_signal += 0.05
    multi_leave_signal = _clamp01(multi_leave_signal)

    chunk_size = len(chunk)
    unique_players = set()
    for hand in chunk:
        unique_players.update(_player_uids(hand))
    unique_player_count = len(unique_players)

    bonus = 0.0
    if chunk_size >= 100 and filtered_multi_leave <= 1:
        bonus += 0.15
    if unique_player_count <= 16:
        bonus += 0.1

    combined_score = 0.65 * multi_leave_signal + 0.35 * avg_score + bonus
    return round(_clamp01(combined_score), 6)


def _score_filtered_zero(chunk: List[dict], multi_joinleave: int, raw_multi_leave: int) -> float:
    """
    Score filtered=0 chunk using ML model if available, fallback to heuristics.
    ML model provides 99.67% accuracy vs 86.38% for pure heuristics.
    Uses ML regardless of multi_signal status - applies to ALL filtered=0.
    """
    global _ML_REQUEST_F0_ML_USED, _ML_REQUEST_F0_HEUR_FALLBACK

    # Try ML first (lazy load on first call) - applies to ALL filtered=0 chunks
    _load_ml_model_filtered0()
    if _ML_FILTERED0_MODEL is not None and _ML_FILTERED0_SCALER is not None:
        ml_score = _score_filtered_zero_ml(chunk)
        if ml_score is not None:
            _ML_REQUEST_F0_ML_USED += 1
            return round(_clamp01(ml_score), 6)
    
    # Fallback to heuristics if ML unavailable
    _ML_REQUEST_F0_HEUR_FALLBACK += 1
    weights = FILTERED0_WEIGHTS
    score = 1.0

    multi_signal = max(multi_joinleave, raw_multi_leave)
    if multi_signal > 0:
        score -= min(1.0, weights["multi_penalty"] + weights["multi_step"] * max(0, multi_signal - 1))
    else:
        streets_avg = _avg_streets_per_hand(chunk)
        if streets_avg <= weights["street_floor"]:
            street_penalty = 0.0
        else:
            street_penalty = min(1.0, (streets_avg - weights["street_floor"]) / weights["street_span"])
        score -= weights["street_weight"] * street_penalty

    filled_ratio = _avg_filled_seat_ratio(chunk)
    if filled_ratio < weights["filled_threshold"]:
        score += min(
            weights["filled_boost"],
            (weights["filled_threshold"] - filled_ratio) * weights["filled_scale"],
        )

    avg_players = _avg_players_per_hand_chunk(chunk)
    if avg_players < weights["players_threshold"]:
        score += min(
            weights["players_boost"],
            (weights["players_threshold"] - avg_players) * weights["players_scale"],
        )

    # NEW: Action-based signals (from legacy scoring, now applied to filtered=0)
    # These are only applied if their weights are non-zero to avoid computation waste
    if any(weights.get(k, 0) > 0 for k in ["action_aggressiveness_weight", "call_ratio_weight", "fold_ratio_weight", "showdown_freq_weight", "street_variance_weight", "player_volatility_weight"]):
        call_ratio, check_ratio, fold_ratio, raise_ratio = _action_ratios_chunk(chunk)
        
        # High call ratio suggests bot grinding (penalize)
        if call_ratio > 0.35:
            score -= weights.get("call_ratio_weight", 0) * _clamp01((call_ratio - 0.35) / 0.35)
        
        # Low fold ratio suggests bot (bots fold less) - penalize
        if fold_ratio < 0.55:
            score -= weights.get("fold_ratio_weight", 0) * _clamp01((0.55 - fold_ratio) / 0.55)
        
        # Raise/Call ratio: low aggression suggests bot - penalize
        if raise_ratio > 0 and call_ratio > 0:
            aggressiveness = raise_ratio / (call_ratio + raise_ratio + 1e-6)
            if aggressiveness < 0.20:
                score -= weights.get("action_aggressiveness_weight", 0) * _clamp01((0.20 - aggressiveness) / 0.20)
        
        # Lower showdown frequency suggests bot tight play (penalize)
        showdown_freq = _showdown_frequency(chunk)
        if showdown_freq < 0.35:
            score -= weights.get("showdown_freq_weight", 0) * _clamp01((0.35 - showdown_freq) / 0.35)
        
        # Street variance: humans have more variance (reward)
        street_var = _street_variance(chunk)
        if street_var > 0.5:
            score += weights.get("street_variance_weight", 0) * _clamp01((street_var - 0.5) / 1.0)
        
        # Player volatility: humans have more variance (reward)
        player_vol = _player_volatility(chunk)
        if player_vol > 0.8:
            score += weights.get("player_volatility_weight", 0) * _clamp01((player_vol - 0.8) / 1.5)

    return round(_clamp01(score), 6)


def _score_filtered_one(chunk: List[dict], multi_joinleave: int, raw_multi_leave: int) -> float:
    global _ML_REQUEST_F1_ML_USED, _ML_REQUEST_F1_HEUR_FALLBACK
    global _ML_REQUEST_F1_HARDCUT_FORCED_HUMAN

    # Hard cutoff — always retained regardless of ML availability
    if multi_joinleave > 1 or raw_multi_leave > 1:
        _ML_REQUEST_F1_HARDCUT_FORCED_HUMAN += 1
        return 0.0

    # Try ML model first
    _load_ml_model_filtered1()
    if _ML_FILTERED1_MODEL is not None and _ML_FILTERED1_SCALER is not None:
        ml_score = _score_filtered_one_ml(chunk)
        if ml_score is not None:
            _ML_REQUEST_F1_ML_USED += 1
            return round(_clamp01(ml_score), 6)

    # Fallback to heuristic weights if ML unavailable
    _ML_REQUEST_F1_HEUR_FALLBACK += 1
    weights = FILTERED1_WEIGHTS
    score = 1.0

    streets_avg = _avg_streets_per_hand(chunk)
    if streets_avg <= weights["street_floor"]:
        street_penalty = 0.0
    else:
        street_penalty = min(1.0, (streets_avg - weights["street_floor"]) / weights["street_span"])
    score -= weights["street_weight"] * street_penalty

    filled_ratio = _avg_filled_seat_ratio(chunk)
    if filled_ratio < weights["filled_threshold"]:
        score += min(
            weights["filled_boost"],
            (weights["filled_threshold"] - filled_ratio) * weights["filled_scale"],
        )

    avg_players = _avg_players_per_hand_chunk(chunk)
    if avg_players < weights["players_threshold"]:
        score += min(
            weights["players_boost"],
            (weights["players_threshold"] - avg_players) * weights["players_scale"],
        )

    return round(_clamp01(score), 6)


def chunk_payload_is_legacy(chunk: List[dict]) -> bool:
    """Detect whether chunk comes from the old fully-sanitized payloads."""
    if not chunk:
        return True

    modern_hands = 0
    total_hands = 0
    for hand in chunk:
        if not isinstance(hand, dict):
            continue
        total_hands += 1
        if _hand_has_modern_actions(hand):
            modern_hands += 1
            if modern_hands >= 1:
                return False

    if total_hands == 0:
        return True
    return True


def score_chunk(chunk: List[dict]) -> float:
    """Backward-compatible helper used by older code paths."""
    return score_chunk_modern(chunk)


# ---------------------------------------------------------------------------
# gen7heur1 – chunk-level heuristic scorer based on benchmark-derived profile
# ---------------------------------------------------------------------------

_GEN7HEUR1_PROFILE: Optional[dict] = None
_GEN7HEUR1_PROFILE_LOCK = False  # simple re-entrancy guard

_GEN7HEUR1_STANDARD_ACTIONS: Set[str] = {"bet", "call", "check", "fold", "raise"}

_EPS_G7 = 1e-9


def _load_gen7heur1_profile() -> dict:
    global _GEN7HEUR1_PROFILE
    if _GEN7HEUR1_PROFILE is not None:
        return _GEN7HEUR1_PROFILE
    env_path = os.getenv("POKER44_GEN7HEUR9_PROFILE", "")
    if env_path:
        profile_path = Path(env_path)
    else:
        profile_path = Path(__file__).resolve().parents[1] / "models" / "benchmark_heuristic_profile_gen7heur9.json"
    import json as _json
    with open(profile_path, "r", encoding="utf-8") as _f:
        _GEN7HEUR1_PROFILE = _json.load(_f)
    return _GEN7HEUR1_PROFILE


def _gen7heur1_extract_features(chunk: List[dict]) -> Dict[str, float]:
    """Extract the same 24 chunk-level features used in the gen7heur1 profile.

    'other' action types are skipped entirely, matching the production-data
    distribution where both bots and humans generate other-type actions.
    """
    action_counter: Counter = Counter()
    street_counter: Counter = Counter()
    actor_counter: Counter = Counter()

    actions_per_hand: List[float] = []
    raise_bb_values: List[float] = []
    bet_bb_values: List[float] = []
    pot_values: List[float] = []
    player_counts: List[float] = []
    street_depths: List[float] = []

    showdown_count = 0

    for hand in chunk:
        actions = hand.get("actions") or []
        outcome = hand.get("outcome") or {}
        players = hand.get("players") or []
        streets = hand.get("streets") or []

        actions_per_hand.append(float(len(actions)))
        player_counts.append(float(len(players)))
        street_depths.append(float(len(streets)))
        pot_values.append(float(outcome.get("total_pot") or 0.0))

        if bool(outcome.get("showdown")):
            showdown_count += 1

        for action in actions:
            atype = str(action.get("action_type") or "other")
            if atype not in _GEN7HEUR1_STANDARD_ACTIONS:
                continue  # skip 'other' and any unknown types
            street = str(action.get("street") or "unknown")
            actor = str(action.get("actor_seat") or "?")

            action_counter[atype] += 1
            street_counter[street] += 1
            actor_counter[actor] += 1

            bb_size = float(action.get("normalized_amount_bb") or 0.0)
            if atype == "raise":
                raise_bb_values.append(bb_size)
            elif atype == "bet":
                bet_bb_values.append(bb_size)

    hand_count = len(chunk)
    total_actions = sum(action_counter.values())
    denom = max(1, total_actions)

    def _smean(vals: List[float]) -> float:
        return float(sum(vals) / len(vals)) if vals else 0.0

    def _sstd(vals: List[float]) -> float:
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return float(math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)))

    def _entropy(counter: Counter) -> float:
        t = sum(counter.values())
        if t <= 0:
            return 0.0
        return float(-sum((c / t) * math.log(c / t + _EPS_G7) for c in counter.values()))

    return {
        "action_entropy": _entropy(action_counter),
        "actions_per_hand_mean": _smean(actions_per_hand),
        "actions_per_hand_std": _sstd(actions_per_hand),
        "actions_total": float(total_actions),
        "actor_entropy": _entropy(actor_counter),
        "aggression_ratio": (
            (action_counter.get("raise", 0) + action_counter.get("bet", 0))
            / max(1, action_counter.get("call", 0) + action_counter.get("check", 0))
        ),
        "bet_bb_mean": _smean(bet_bb_values),
        "bet_bb_std": _sstd(bet_bb_values),
        "bet_ratio": action_counter.get("bet", 0) / denom,
        "call_ratio": action_counter.get("call", 0) / denom,
        "check_ratio": action_counter.get("check", 0) / denom,
        "chunk_size": float(hand_count),
        "fold_ratio": action_counter.get("fold", 0) / denom,
        "players_mean": _smean(player_counts),
        "players_std": _sstd(player_counts),
        "pot_mean": _smean(pot_values),
        "pot_std": _sstd(pot_values),
        "raise_bb_mean": _smean(raise_bb_values),
        "raise_bb_std": _sstd(raise_bb_values),
        "raise_ratio": action_counter.get("raise", 0) / denom,
        "showdown_rate": showdown_count / max(1, hand_count),
        "street_depth_mean": _smean(street_depths),
        "street_depth_std": _sstd(street_depths),
        "street_entropy": _entropy(street_counter),
    }


def _sigmoid_g7(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def score_chunk_gen7heur1(chunk: List[dict]) -> Tuple[float, str]:
    """Score a chunk using the gen7heur1 benchmark-derived heuristic profile.

    Returns (risk_score, route_label) where risk_score is in [0, 1].
    0 = human, 1 = bot.
    """
    if not chunk:
        return 0.5, "gen7heur1_empty"

    try:
        profile = _load_gen7heur1_profile()
    except Exception:
        return 0.5, "gen7heur1_profile_load_error"

    features = _gen7heur1_extract_features(chunk)
    weights = profile["weights"]
    stats = profile["class_stats"]

    raw = 0.0
    for feat in profile["feature_names"]:
        w = float(weights.get(feat, 0.0))
        if w == 0.0:
            continue
        mu_h = float(stats["human"][feat]["mean"])
        mu_b = float(stats["bot"][feat]["mean"])
        sd_h = float(stats["human"][feat]["std"])
        sd_b = float(stats["bot"][feat]["std"])
        midpoint = 0.5 * (mu_h + mu_b)
        pooled = math.sqrt((sd_h * sd_h + sd_b * sd_b) / 2.0) + _EPS_G7
        z = (float(features.get(feat, midpoint)) - midpoint) / pooled
        raw += w * z

    risk = _sigmoid_g7(raw)

    # Shrink confidence for smaller chunks
    smin = float(profile["score_logic"].get("chunk_size_min", 40))
    smax = float(profile["score_logic"].get("chunk_size_max", 80))
    cmin = float(profile["score_logic"].get("chunk_confidence_min", 0.65))
    cmax = float(profile["score_logic"].get("chunk_confidence_max", 1.0))
    size = float(features.get("chunk_size", smin))
    alpha = max(0.0, min(1.0, (size - smin) / max(_EPS_G7, smax - smin)))
    confidence = cmin + (cmax - cmin) * alpha

    score = 0.5 + (risk - 0.5) * confidence
    return round(max(0.0, min(1.0, score)), 6), "gen7heur1"


def score_chunk_gen7heur9(chunk: List[dict]) -> Tuple[float, str]:
    """Score a chunk with the gen7heur9 profile (same math as gen7heur1)."""
    score, route = score_chunk_gen7heur1(chunk)
    if route == "gen7heur1":
        return score, "gen7heur9"
    return score, route.replace("gen7heur1", "gen7heur9")




def get_chunk_scorer_startup_check(scorer: str) -> Dict[str, object]:
    """Return startup readiness diagnostics for chunk-level scorers.

    This is intended for miner startup logging so operators can detect missing
    artifacts before the first request arrives.
    """
    scorer_norm = (scorer or "").strip().lower()
    info: Dict[str, object] = {
        "scorer": scorer_norm,
        "active": scorer_norm in {"gen7heur9"},
        "ok": True,
        "error": None,
        "details": {},
    }

    if scorer_norm in {"gen7heur9"}:
        env_path = os.getenv("POKER44_GEN7HEUR9_PROFILE", "")
        profile_path = (
            Path(env_path)
            if env_path
            else Path(__file__).resolve().parents[1] / "models" / "benchmark_heuristic_profile_gen7heur9.json"
        )
        details = {
            "profile_path": str(profile_path),
            "profile_exists": profile_path.exists(),
            "rebalance_target": "50/50" if scorer_norm == "gen7heur2" else "disabled",
        }
        info["details"] = details
        try:
            _load_gen7heur1_profile()
        except Exception as exc:
            info["ok"] = False
            info["error"] = str(exc)

    return info
