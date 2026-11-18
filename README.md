# slack-agent-controller

Run a Claude agent on your own Linux server and drive it from Slack. You can watch it work, approve
its actions with buttons, hand sessions back and forth between Slack and your terminal, and have
long computations reported back to you, all from your phone.

[中文说明 / Chinese README](README.zh-CN.md)

> slack-agent-controller is an independent open-source project, *powered by Claude* through the
> [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview). It is not made or endorsed
> by Anthropic. Using the Agent SDK is subject to
> [Anthropic's Commercial Terms](https://www.anthropic.com/legal/commercial-terms); you need your own
> Anthropic API key or Amazon Bedrock access.

**Read [SECURITY.md](SECURITY.md) before you deploy it.** This bot lets people in your Slack
workspace run code on your machine. Its safeguards reduce that risk, but they do not remove it.

## What it does

- **Every Slack thread is one agent session.** Messages in the thread continue the session. Sessions
  are saved to disk and resume after restarts. Replies stream into Slack, and every tool call is
  shown.
- **Approvals with buttons.** Commands, file edits and web requests show a card with
  Approve / Deny / Approve-all-for-this-session buttons, plus "Run & remember here" for
  commands that are safe to repeat. Unanswered cards are denied after 30 minutes. There are two
  modes per thread:
  - `ask`: you approve every action that has side effects;
  - `auto`: Claude Code's auto-mode classifier approves routine actions, but protected files,
    credentials, environment dumps and network transfers still come to Slack.
- **Hard limits that approvals cannot override.** Secrets and config paths (`~/.ssh`, `~/.aws`,
  `.env`, `*.pem`, the bot's own files, ...) are blocked, also when a shell wildcard or brace
  expansion would reach them. Commands that reference credentials are denied. Work is confined to
  the directories you allow.
- **Built for phones.**
  - Buttons under each reply: Stop, Hand to terminal, Switch mode, Status.
  - "Interrupt and send now" for a message you send while the agent is busy.
  - Claude's clarifying questions appear as tappable choices.
  - Screenshots, logs and PDFs you attach are passed to Claude.
  - A Slack Home tab shows what needs you, what is running, jobs and GPUs.
- **Long computations (`sac-job`).** Training runs and simulations run as their own systemd user
  units:
  - free GPUs are picked automatically, and jobs queue when none are free;
  - credentials are stripped from the job environment;
  - progress, errors and the result are posted in the thread;
  - each job has a Cancel button.
  Jobs keep running if the agent or the bot restarts.
- **Terminal and Slack together (optional).** Sessions you run in a terminal can be mirrored into
  Slack: one message per turn, plus a full Markdown log. Slack waits while the terminal has the
  session open, and you can continue in Slack once you exit. `!exit` hands a Slack session back to
  your terminal.
- **Housekeeping.**
  - An audit log of every tool decision.
  - Per-turn cost limits, and a cost estimate for each session.
  - A restart report: messages that were cut off by a bot restart are reported, never replayed.
  - Stall notices, project shortcuts, `!import` / `!fork` of sessions, and per-thread model and
    effort.

## How it works

```
Slack (Socket Mode, no public URL) ──► slack-agent-controller (this repo, one process)
                                          │  one Claude Agent SDK client per thread
                                          ▼
                                   Claude Code CLI (bundled with the SDK)
                                          │  every tool call -> PreToolUse hook -> policy
                                          │  needs approval  -> Slack card -> your tap
                                          ▼
                                   your files, shell, GPUs  (inside CWD_ROOTS)
```

Settings files (`~/.claude/settings.json`) are deliberately **not** loaded into the bot's sessions,
because their allow-rules would silently bypass the Slack approvals. The bot passes what it needs
(model, effort, memory directory) explicitly.

## Requirements

| | |
|---|---|
| OS | **Linux** for everything. The core bot also runs on macOS. Terminal mirroring needs Linux; `sac-job` needs systemd user services (`loginctl enable-linger $USER` so they outlive your logins). |
| Python | 3.11 or newer |
| Model access | **Amazon Bedrock** (API key, profile or access keys) or an **Anthropic API key**. claude.ai login is not supported: Anthropic does not allow it for third-party tools. Auto mode needs a model that supports it. |
| Slack | A workspace where you can create apps (you need to be able to install one). |
| Optional | `nvidia-smi` (GPU status and GPU jobs), `squeue` (Slurm status) |

## Quick start

1. **Create the Slack app.** At api.slack.com/apps choose *Create New App → From a manifest* and
   paste [`slack_manifest.yaml`](slack_manifest.yaml). Install it to your workspace. Then:
   - copy the *Bot User OAuth Token* (`xoxb-…`);
   - under *Basic Information → App-Level Tokens*, create a token with scope `connections:write`
     (`xapp-…`);
   - find your member ID (profile → ⋯ → *Copy member ID*).
2. **Install.**
   ```bash
   git clone https://github.com/tianhao0804/slack-agent-controller ~/slack-agent-controller
   cd ~/slack-agent-controller
   python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```
3. **Configure.**
   ```bash
   mkdir -p ~/.config/slack-agent-controller
   cp .env.example ~/.config/slack-agent-controller/env
   chmod 600 ~/.config/slack-agent-controller/env
   $EDITOR ~/.config/slack-agent-controller/env   # tokens, your member ID, provider, CWD_ROOTS
   .venv/bin/python bot.py --check
   ```
4. **Run it.** For a first test, run `.venv/bin/python bot.py` and DM the bot. To keep it running:
   ```bash
   mkdir -p ~/.config/systemd/user
   cp deploy/slack-agent-controller.service ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now slack-agent-controller
   loginctl enable-linger "$USER"     # keep user services running when you are logged out
   ```
   The unit assumes the repository is at `~/slack-agent-controller`; edit the paths if it is not.
5. **Optional extras.**
   - Terminal mirroring: set `MIRROR_DIRS`, `MIRROR_CHANNEL` and `MIRROR_USER`, then run
     `.venv/bin/python deploy/install_hooks.py`. This adds hooks to `~/.claude/settings.json`; a
     backup is made first, and `--remove` undoes it.
   - Jobs from your shell: `ln -s ~/slack-agent-controller/deploy/sac-job ~/bin/sac-job`.

## Using it

DM the bot, or post in a private channel listed in `ALLOWED_CHANNEL_IDS`. The first message starts a
session; replies in its thread continue it.

| Command | What it does |
|---|---|
| `!new <dir\|alias> [message]` | New session in a directory (must be under `CWD_ROOTS`) or a `PROJECTS` alias |
| `!projects` | Your project shortcuts, each with a *New session* button |
| `!sessions [all]` | Sessions with titles, state and *Open* buttons (`all`: also recent terminal sessions) |
| `!stop` · `!stop bg` · `!stop all` | Stop the running reply and drop queued messages · stop background tasks · both |
| `!status` | GPUs, Slurm queue, this session's id, tokens, cost, model, mode |
| `!mode [ask\|auto]` | Permission mode of this thread |
| `!model [...]` · `!effort [...]` | Model and effort of this thread (no argument: show current and last used) |
| `!allowlist [remove <id>\|clear]` | Commands you saved with *Run & remember here* |
| `!jobs` | Jobs started with `sac-job` |
| `!upload <path>` | Attach a file from the session's directory |
| `!resume <id>` · `!import <id…>` · `!fork` | Attach a terminal session · import sessions with their last exchanges · branch this session |
| `!exit` | Close this thread's agent process so the session can be used in a terminal |
| `!end` | End this session and unbind the thread (the transcript is kept) |
| `!unlock` | Release a stale terminal lock |
| `!help` | All commands |

**Files.** Attach screenshots, logs, PDFs or notebooks to a message (with or without text). They
are downloaded to `~/.cache/slack-agent-controller/inbox/`, outside your projects, and deleted after
7 days. Claude reads them with its Read tool.

**Jobs.** Ask the agent to "run X as a job on 2 GPUs", or run this yourself:

```bash
sac-job submit --name relax --gpus 2 -- python relax.py --cfg run.yaml
sac-job list | status <id> | log <id> -n 100 | cancel <id>
```

**Memory.** Claude Code's auto-memory is enabled, but each session may only read and write its own
project's memory directory (`~/.claude/projects/<project>/memory/`). That is the same folder your
terminal sessions in that directory use. The rest of `~/.claude` stays off-limits.

Configuration reference: [`.env.example`](.env.example) (every option is commented).

## Files and logs

| Path | Contents |
|---|---|
| `~/.config/slack-agent-controller/env` | Configuration and secrets (mode 600) |
| `~/.local/state/slack-agent-controller/` | `sessions.json` (thread ↔ session), `audit.jsonl` (every tool decision, rotated), `bot.log`, `allowlist.json`, `tasks.json`, `mirror.json`; the agent cannot read this directory |
| `~/.local/share/slack-agent-controller/jobs/<id>/` | `sac-job` records and logs |
| `~/.cache/slack-agent-controller/inbox/` | Files you attached in Slack |
| `journalctl --user -u slack-agent-controller -f` | Service log |

## Troubleshooting

| Symptom | Check |
|---|---|
| `config error: … chmod 600` | `chmod 600 ~/.config/slack-agent-controller/env` |
| The bot does not answer DMs | Your member ID is in `ALLOWED_SLACK_USER_IDS`; `messages_tab_read_only_enabled: false` in the manifest; `journalctl --user -u slack-agent-controller` |
| Buttons do nothing | Interactivity is enabled in the app settings |
| "missing the `files:read` permission" | Add the scope under *OAuth & Permissions* and reinstall the app |
| Home tab is empty | Enable *Home Tab* and the `app_home_opened` event, then reinstall the app |
| "another slack-agent-controller is already running" | Only one instance per user and machine; stop the other one |
| Authentication errors | The provider key expired: update it in the env file and restart the service |

## Development

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/ruff check slack_agent_controller tests deploy bot.py
.venv/bin/mypy --ignore-missing-imports slack_agent_controller bot.py
.venv/bin/pytest -q          # offline tests: fake Slack, no model calls
```

The integration tests (`tests/integration_*.py`) use real model calls and a fake Slack. Run them one
at a time, for example `python -m tests.integration_fake_slack`. They need provider credentials in
the environment and cost a few cents each. See [CONTRIBUTING.md](CONTRIBUTING.md).

**Compatibility.** Tested with `claude-agent-sdk` 0.2.182, which bundles Claude Code 2.1.325. A few
features depend on Claude Code internals that may change between releases: transcript format, hook
payloads, background-session processes, and the `autoMemoryDirectory` setting. Pin the SDK version,
and re-run the integration tests before you upgrade.

## License

[MIT](LICENSE).
