#!/bin/bash

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "Użycie: $0 ID1,ID2,ID3... [ML_MAX_HANDS] [REMOVE_OTHER] [MODEL]"
  echo "ML_MAX_HANDS (opcjonalnie): maks. liczba rąk dla routingu single-hand ML vote (domyślnie 40)"
  echo "REMOVE_OTHER (opcjonalnie): jeśli ustawione na 1/true, usuwa 'other' akcje z rąk (domyślnie 0)"
  echo "MODEL (opcjonalnie): gen7heur5"
  exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDS_STRING="$1"
ML_MAX_HANDS="${2:-40}"
REMOVE_OTHER="${3:-0}"
MODEL="${4:-}"
ENV_FILE="${ENV_FILE:-$REPO/.env}"

SINGLE_HAND_MODEL_ALIAS="gen7heur5"
SINGLE_HAND_MODEL_PATH="weights/ml_single_hand_model.pkl"
SINGLE_HAND_SCALER_PATH="weights/ml_single_hand_scaler.pkl"
CHUNK_SCORER="gen7heur5"

# Public manifest pinning (release repo snapshot used for transparency checks).
MANIFEST_REPO_URL="${POKER44_MODEL_REPO_URL:-https://github.com/tomkaba/poker44-miner-gen7heur5}"
MANIFEST_REPO_COMMIT="${POKER44_MODEL_REPO_COMMIT:-$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)}"
MANIFEST_IMPL_FILES="${POKER44_MODEL_IMPLEMENTATION_FILES:-neurons/miner.py,poker44/miner_heuristics.py}"
MANIFEST_IMPL_SHA256="${POKER44_MODEL_IMPLEMENTATION_SHA256:-0c5c6d4130ece8d455d0b9c0f72acafa2c9f5849900e46e3d3681be5aaaaace7}"

case "$MODEL" in
  "" )
    ;;
  "gen7heur5" )
    ;;
  * )
    echo "ERROR: Niepoprawny MODEL='$MODEL'. Dozwolone: gen7heur5"
    exit 1
    ;;
esac

if ! [[ "$ML_MAX_HANDS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: ML_MAX_HANDS musi być liczbą całkowitą >= 1, otrzymano '$ML_MAX_HANDS'"
  exit 1
fi
if [[ "$ML_MAX_HANDS" -lt 1 ]]; then
  echo "ERROR: ML_MAX_HANDS musi być >= 1, otrzymano '$ML_MAX_HANDS'"
  exit 1
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "[env] Loaded $ENV_FILE"
else
  echo "[env] File not found, skipping: $ENV_FILE"
fi

# Process IDs one by one
for raw_id in $(echo "$IDS_STRING" | tr ',' '\n'); do
  # Trim whitespace
  I=$(echo "$raw_id" | tr -d ' ')
  
  if [[ -z "$I" ]]; then
    continue
  fi
  
  if ! [[ "$I" =~ ^[0-9]+$ ]]; then
    echo "WARN: Invalid ID '$I', skipping"
    continue
  fi

  PORT=$((12080 + I))
  SESSION="sn126b_m${I}"

  echo "[start] ID=$I SESSION=$SESSION PORT=$PORT"
  echo "[start] ML_MAX_HANDS=$ML_MAX_HANDS"
  echo "[start] REMOVE_OTHER=$REMOVE_OTHER"
  echo "[start] SINGLE_HAND_MODEL_ALIAS=$SINGLE_HAND_MODEL_ALIAS"
  echo "[start] SINGLE_HAND_MODEL_PATH=$SINGLE_HAND_MODEL_PATH"
  echo "[start] SINGLE_HAND_SCALER_PATH=$SINGLE_HAND_SCALER_PATH"
  echo "[start] MANIFEST_REPO_URL=$MANIFEST_REPO_URL"
  echo "[start] MANIFEST_REPO_COMMIT=$MANIFEST_REPO_COMMIT"
  echo "[start] MANIFEST_IMPL_FILES=$MANIFEST_IMPL_FILES"
  echo "[start] MANIFEST_IMPL_SHA256=$MANIFEST_IMPL_SHA256"

  # Kill old session if exists (exact match via grep)
  echo "[cleanup] Attempting to quit old session: $SESSION"
  OLD_PID=$(screen -list 2>/dev/null | grep "\.$SESSION[[:space:]]" | awk '{print $1}' | cut -d. -f1)
  if [[ -n "$OLD_PID" ]]; then
    screen -S "$OLD_PID" -X quit 2>/dev/null || true
    echo "[cleanup] Killed PID=$OLD_PID"
  fi
  
  sleep 1

  # Start new session
  echo "[launch] Creating screen session $SESSION..."
  
  screen -dmS "$SESSION" /bin/bash -c "
    cd $REPO
    source .venv/bin/activate
    export PYTHONPATH=$REPO:\${PYTHONPATH:-}
    export ML_MAX_HANDS=$ML_MAX_HANDS
    export REMOVE_OTHER=$REMOVE_OTHER
    export POKER44_SINGLE_HAND_MODEL_ALIAS=$SINGLE_HAND_MODEL_ALIAS
    export POKER44_SINGLE_HAND_MODEL_PATH=$SINGLE_HAND_MODEL_PATH
    export POKER44_SINGLE_HAND_SCALER_PATH=$SINGLE_HAND_SCALER_PATH
    export POKER44_CHUNK_SCORER=$CHUNK_SCORER
    export POKER44_MODEL_REPO_URL=$MANIFEST_REPO_URL
    export POKER44_MODEL_REPO_COMMIT=$MANIFEST_REPO_COMMIT
    export POKER44_MODEL_IMPLEMENTATION_FILES=$MANIFEST_IMPL_FILES
    export POKER44_MODEL_IMPLEMENTATION_SHA256=$MANIFEST_IMPL_SHA256
    echo '[routing] ML_MAX_HANDS='$ML_MAX_HANDS
    echo '[routing] REMOVE_OTHER='$REMOVE_OTHER
    echo '[routing] SINGLE_HAND_MODEL_ALIAS='$SINGLE_HAND_MODEL_ALIAS
    echo '[routing] SINGLE_HAND_MODEL_PATH='$SINGLE_HAND_MODEL_PATH
    echo '[routing] SINGLE_HAND_SCALER_PATH='$SINGLE_HAND_SCALER_PATH
    echo '[routing] CHUNK_SCORER='$CHUNK_SCORER
    echo '[manifest] POKER44_MODEL_REPO_URL='$MANIFEST_REPO_URL
    echo '[manifest] POKER44_MODEL_REPO_COMMIT='$MANIFEST_REPO_COMMIT
    echo '[manifest] POKER44_MODEL_IMPLEMENTATION_FILES='$MANIFEST_IMPL_FILES
    echo '[manifest] POKER44_MODEL_IMPLEMENTATION_SHA256='$MANIFEST_IMPL_SHA256
    .venv/bin/python -m neurons.miner \\
      --netuid 126 \\
      --wallet.name sn126b \\
      --wallet.hotkey hk$I \\
      --subtensor.network finney \\
      --axon.port $PORT \\
      --logging.debug
    echo '[miner-exit] Process ended, shell remains active'
    /bin/bash
  "

  RC=$?
  if [ $RC -eq 0 ]; then
    echo "[✔] Session $SESSION created successfully"
  else
    echo "[✗] Failed to create session $SESSION (rc=$RC)"
  fi

  sleep 0.5
done

echo "[done] All sessions requested"
