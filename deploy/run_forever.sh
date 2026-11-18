#!/usr/bin/env bash
# Fallback supervisor when systemd --user is unavailable: run inside tmux.
#   tmux new-session -d -s slack-agent-controller ~/slack-agent-controller/deploy/run_forever.sh
# Reboot persistence (crontab -e):
#   @reboot tmux new-session -d -s slack-agent-controller $HOME/slack-agent-controller/deploy/run_forever.sh
set -u
umask 077
cd "$(dirname "$0")/.." || exit 1
while true; do
  .venv/bin/python bot.py --env-file "$HOME/.config/slack-agent-controller/env"
  rc=$?
  if [ "$rc" -eq 2 ]; then echo "config error; not restarting" >&2; exit 2; fi
  echo "bot exited with $rc; restarting in 15s" >&2
  sleep 15
done
