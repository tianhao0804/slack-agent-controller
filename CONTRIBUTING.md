# Contributing

Thanks for helping. Bug reports, fixes and new features are all welcome. For anything large, please
open an issue first so we can agree on the approach.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
```

## Checks (run before every pull request)

```bash
.venv/bin/ruff check slack_agent_controller tests deploy bot.py
.venv/bin/mypy --ignore-missing-imports slack_agent_controller bot.py
.venv/bin/pytest -q
```

The `pytest` suite is offline. It uses a fake Slack client, temporary directories and no model
calls, and CI runs it on every push.

## Integration tests

`tests/integration_*.py` drive the real Claude Code CLI through the Agent SDK, with a fake Slack.
They need provider credentials in your environment, for example `CLAUDE_CODE_USE_BEDROCK=1` and
`AWS_REGION` plus `AWS_BEARER_TOKEN_BEDROCK`, or `ANTHROPIC_API_KEY`. Most use Haiku and cost a few
cents. Run them one at a time:

```bash
.venv/bin/python -m tests.integration_fake_slack   # approvals, streaming, stop, restart/resume, cost
.venv/bin/python -m tests.integration_mode         # ask vs auto mode
.venv/bin/python -m tests.integration_automode     # protected files still ask in auto mode
.venv/bin/python -m tests.integration_product      # questions as buttons, interrupt-and-send
.venv/bin/python -m tests.integration_inbox        # attached images are read
.venv/bin/python -m tests.integration_allowlist    # remembered commands
.venv/bin/python -m tests.integration_memory       # project auto-memory
.venv/bin/python -m tests.integration_terminal     # terminal writes -> Slack reloads (needs `claude` on PATH)
.venv/bin/python -m tests.integration_mirror       # terminal mirroring (Linux, `claude` on PATH)
.venv/bin/python -m tests.integration_jobs         # sac-job with systemd user units (Linux)
```

If your change touches session handling, permissions, mirroring or jobs, please run the matching
integration test and say so in the pull request.

## Guidelines

- **Security first.** A change that relaxes a permission check needs a test showing what it still
  blocks. When you fix a bypass, add the bypass as a regression test (see
  `tests/test_review_regressions.py` and `tests/test_expansion.py`).
- Keep the bot's sessions independent of `~/.claude/settings.json`. Pass whatever is needed through
  `ClaudeAgentOptions`.
- Pass user-visible text through `policy.redact_log` when it may contain tool output.
- Branding: do not call the project or the Slack app "Claude Code" (Anthropic's Agent SDK branding
  guidelines). "Powered by Claude" is fine.

## Adding a model backend

Providers that Claude Code supports, such as Bedrock and the Anthropic API, are configured in
`config.check_provider` and passed to the CLI through the environment. A backend that is not Claude
Code (for example an OpenAI agent) would need its own implementation of the session layer
(`session.ThreadSession`): streaming, the permission hook, transcripts and resume. Please open an
issue to discuss the design first.
