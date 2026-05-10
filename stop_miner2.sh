#!/bin/bash

# Sprawdzenie argumentu
if [ -z "$1" ]; then
  echo "Użycie: $0 ID1,ID2,ID3..."
  exit 1
fi

# Zamiana przecinków na tablicę
IFS=',' read -ra IDS <<< "$1"

for I in "${IDS[@]}"; do
  SESSION="sn126b_m${I}"

  echo "Zatrzymuję screen: $SESSION"

  # Sprawdź czy istnieje
  if screen -list | grep -q "\.${SESSION}[[:space:]]"; then
    screen -S "$SESSION" -X quit
    echo "✔ Zabity: $SESSION"
  else
    echo "⚠ Nie istnieje: $SESSION"
  fi
done