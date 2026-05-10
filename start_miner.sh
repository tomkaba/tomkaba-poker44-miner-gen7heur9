#!/bin/bash

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "Użycie: $0 ID1,ID2,ID3... [ML_MAX_HANDS] [REMOVE_OTHER] [MODEL]"
  echo "ML_MAX_HANDS (opcjonalnie): maks. liczba rąk dla routingu single-hand ML vote (domyślnie 40)"
  echo "REMOVE_OTHER (opcjonalnie): jeśli ustawione na 1/true, usuwa 'other' akcje z rąk (domyślnie 0)"
  echo "MODEL (opcjonalnie): 4 | 4_17 | 5 | 5_17 (domyślnie: active = weights/ml_single_hand_model.pkl)"
  exit 1
fi

REPO="/home/tk/Poker44-subnet-main"
IDS_STRING="$1"
ML_MAX_HANDS="${2:-40}"
REMOVE_OTHER="${3:-0}"
MODEL="${4:-}"
ENV_FILE="${ENV_FILE:-$REPO/.env}"

SINGLE_HAND_MODEL_ALIAS="active"
SINGLE_HAND_MODEL_PATH="weights/ml_single_hand_model.pkl"
SINGLE_HAND_SCALER_PATH="weights/ml_single_hand_scaler.pkl"

case "$MODEL" in
  "" )
    ;;
  "4" )
    SINGLE_HAND_MODEL_ALIAS="gen4"
    SINGLE_HAND_MODEL_PATH="weights/ml_gen4_model.pkl"
    SINGLE_HAND_SCALER_PATH="weights/ml_gen4_scaler.pkl"
    ;;
  "4_17" )
    SINGLE_HAND_MODEL_ALIAS="gen4_17"
    SINGLE_HAND_MODEL_PATH="weights/ml_gen4_17_model.pkl"
    SINGLE_HAND_SCALER_PATH="weights/ml_gen4_17_scaler.pkl"
    ;;
  "5" )
    SINGLE_HAND_MODEL_ALIAS="gen5"
    SINGLE_HAND_MODEL_PATH="weights/ml_gen5_s123467_model.pkl"
    SINGLE_HAND_SCALER_PATH="weights/ml_gen5_s123467_scaler.pkl"
    ;;
  "5_17" )
    SINGLE_HAND_MODEL_ALIAS="gen5_17"
    SINGLE_HAND_MODEL_PATH="weights/ml_gen5_17_s123467_model.pkl"
    SINGLE_HAND_SCALER_PATH="weights/ml_gen5_17_s123467_scaler.pkl"
    ;;
  * )
    echo "ERROR: Niepoprawny MODEL='$MODEL'. Dozwolone: 4 | 4_17 | 5 | 5_17"
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

  PORT=$((11080 + I))
  SESSION="sn126_m${I}"

  echo "[start] ID=$I SESSION=$SESSION PORT=$PORT"
  echo "[start] ML_MAX_HANDS=$ML_MAX_HANDS"
  echo "[start] REMOVE_OTHER=$REMOVE_OTHER"
  echo "[start] SINGLE_HAND_MODEL_ALIAS=$SINGLE_HAND_MODEL_ALIAS"
  echo "[start] SINGLE_HAND_MODEL_PATH=$SINGLE_HAND_MODEL_PATH"
  echo "[start] SINGLE_HAND_SCALER_PATH=$SINGLE_HAND_SCALER_PATH"

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
    echo '[routing] ML_MAX_HANDS='$ML_MAX_HANDS
    echo '[routing] REMOVE_OTHER='$REMOVE_OTHER
    echo '[routing] SINGLE_HAND_MODEL_ALIAS='$SINGLE_HAND_MODEL_ALIAS
    echo '[routing] SINGLE_HAND_MODEL_PATH='$SINGLE_HAND_MODEL_PATH
    echo '[routing] SINGLE_HAND_SCALER_PATH='$SINGLE_HAND_SCALER_PATH
    .venv/bin/python -m neurons.miner \\
      --netuid 126 \\
      --wallet.name sn126 \\
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
