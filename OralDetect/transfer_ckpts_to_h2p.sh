#!/bin/bash
# Package + transfer trained OralDetect ckpt dirs to Pitt CRC storage as tarballs.
# RUN THIS IN YOUR OWN TERMINAL (DUO 2FA is prompted once; SSH ControlMaster multiplexes the rest).
#   usage: bash transfer_ckpts_to_h2p.sh [run_dir ...]
#   no args -> the currently-STABLE runs. Pass the live ones AFTER they finish training.
set -u
REMOTE=sid51@h2p.crc.pitt.edu
DEST=/ix/lzhan/siyuan/exps/OralDetect_Family/our_ckpts
WD=/ix/lzhan/siyuan/exps/OralDetect_Family/our_ckpts/oraldetect
CP="$HOME/.ssh/cm_h2p_%h_%p_%r"
SSHO=(-o ControlMaster=auto -o "ControlPath=$CP" -o ControlPersist=30m)

RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
  RUNS=(oraldetect_v11dt_s3 oraldetect_oralclipdir oraldetect_v11dt_1024_s1)   # stable now
fi

echo "opening master SSH to $REMOTE (enter DUO once)…"
ssh "${SSHO[@]}" "$REMOTE" "mkdir -p '$DEST' && echo remote-ready: \$(hostname)" || { echo "SSH failed"; exit 1; }

for run in "${RUNS[@]}"; do
  if [ ! -d "$WD/$run" ]; then echo "skip (absent): $run"; continue; fi
  sz=$(du -sh "$WD/$run" | cut -f1)
  echo ">>> streaming $run ($sz) -> $DEST/$run.tar"
  if tar cf - -C "$WD" "$run" | ssh "${SSHO[@]}" "$REMOTE" "cat > '$DEST/$run.tar'"; then
    ssh "${SSHO[@]}" "$REMOTE" "ls -lh '$DEST/$run.tar'"
  else
    echo "!! FAILED: $run (rerun to retry)"
  fi
done

ssh "${SSHO[@]}" -O exit "$REMOTE" 2>/dev/null || true
echo "DONE. verify: ssh $REMOTE 'ls -lh $DEST'"
# to unpack later on the remote:  tar xf oraldetect_v11dt_s3.tar
