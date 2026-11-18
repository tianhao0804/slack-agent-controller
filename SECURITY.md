# Security model

slack-agent-controller gives an AI agent a shell on your machine and lets people in your Slack
workspace direct it. Treat it like remote access to your account.

## Who can do what

- Only Slack users listed in `ALLOWED_SLACK_USER_IDS` are served. This applies to messages and to
  every button click. Everyone else is ignored, logged, and written to the audit log. Any allowlisted
  user can act on any thread, so use this for yourself or a small trusted group.
- The bot reads only DMs and the private channels you list. Use a private channel. The app manifest
  only subscribes to private-channel messages.
- The agent runs as **your Unix user**, with your file permissions.

**If someone takes over your Slack account, they effectively get a shell on your machine.** Enable
two-factor authentication on Slack and keep the allowlist short.

## What the bot enforces

These are evaluated in a `PreToolUse` hook on every tool call. Claude Code settings files are not
loaded, so their allow-rules cannot bypass the checks.

1. **Hard denials.** Approvals cannot override these.
   - Reading or writing protected paths with the file tools: `~/.ssh`, `~/.aws`, `~/.claude`,
     `~/.config/slack-agent-controller`, the bot's state directory, `~/.gnupg`, `~/.netrc`, and
     more. The session's own memory folder is open. In `auto` mode the rest of `~/.claude` (job
     scratch folders, transcripts, other projects' notes) is left to the classifier; Claude Code's
     `settings*.json` and `.credentials.json` stay denied.
   - Protected file names anywhere: `.env`, `*.pem`, `*.key`, `id_rsa*`, `.git-credentials`, and
     more. Add your own with `EXTRA_SENSITIVE_PATHS` and `EXTRA_SENSITIVE_GLOBS`.
   - Shell commands that name provider credentials (`AWS_SECRET_ACCESS_KEY`,
     `AWS_BEARER_TOKEN_BEDROCK`, `ANTHROPIC_API_KEY`, …) or `/proc/*/environ`.
   - Writes outside `CWD_ROOTS`.
2. **Always asked in Slack**, even after "Approve all":
   - shell commands that name protected files, also through wildcards, brace expansion, `~`,
     `$HOME`, `..` or symlinks (`cat .e*` and `cat .{e,}nv` are both caught);
   - commands that may dump the environment (`env`, `printenv`, `os.environ`, …);
   - searches whose scope or glob filter can reach protected files.

   In `ask` mode this covers every protected path and also network transfers (`curl`, `wget`,
   `scp`, …). In `auto` mode only a short list counts, so it feels like auto mode in the terminal:
   secrets (`~/.ssh`, `~/.aws`, `.env`, `*.pem`, the bot's env file, …), files that control
   permissions (Claude Code `settings*.json` and `.credentials.json`, `~/.claude.json`, the bot's
   state directory), directories that hold them (`cd ~/.claude && cat settings.json`), and
   environment dumps. Network transfers and the rest of `~/.claude` go to the classifier, as in
   the terminal.
3. **Everything else** with side effects needs a button tap in `ask` mode. In `auto` mode it goes to
   Claude Code's auto-mode classifier.
   - The classifier treats an explicit request in your prompt as permission. That is why the checks
     in 1 and 2 are done by the bot and not left to the classifier.
   - Read-only tools (Read, Grep, Glob) inside your directories run without asking.
4. **Remembered commands.** "Run & remember here" stores the exact command text together with its
   directory. Some commands are never remembered:
   - anything flagged in 2;
   - anything that deletes, moves, kills or changes permissions;
   - anything whose meaning depends on shell expansion (`$VAR`, `*`, backticks).

   A remembered entry pins the command text, not the scripts it runs.
5. **Output.** Replies, cards, attachments and mirrored logs pass through a redaction filter. It
   catches known key formats (Slack, AWS, Anthropic, Bedrock, GitHub, private keys) and `key=value`
   secrets. Tool outputs of calls that touched protected paths are left out of mirrored logs.
6. **Jobs** (`sac-job`) run with your environment minus anything that looks like a credential
   (`*TOKEN*`, `*SECRET*`, `*KEY*`, `*PASSWORD*`, `AWS_*`, `ANTHROPIC_*`, `SLACK_*`, the SSH agent).
   They can only start inside `CWD_ROOTS`. Submitting or cancelling a job is an approved shell
   command like any other.
7. **Secrets at rest.** The env file must be mode 600 and owned by you, or the bot refuses to start.
   The Slack tokens are removed from the process environment after loading, so the agent never sees
   them. State files are mode 600 in a 700 directory.
8. **Audit.** `audit.jsonl` records every tool decision: time, session, tool, a redacted argument
   summary, the decision, who approved it, and how long that took.

## Known limits (read these)

- **Shell commands can see the model provider's key.** Claude Code passes its environment to the
  shell, and the bot needs the key there to run. Commands that name the key are denied, and commands
  that could dump the environment always ask. But a command can reach the key indirectly, for example
  through a script you approve. Claude Code's environment scrubbing (`SCRUB_SUBPROCESS_ENV=1`)
  closes this gap where its sandbox (bubblewrap + socat) is available.
- **The checks are pattern-based, not an operating-system sandbox.** Indirect references
  (`cat $(echo .env)`, `bash -c '…'`, a script that reads a secret) and recursive searches over a tree
  that contains secrets are not detected. They rely on your approval and on the classifier. If you
  need a hard boundary, run the bot in a container or VM, or as a dedicated user that only has
  access to what it needs.
- **Redaction is best effort.** Unknown secret formats can reach Slack. Everything the bot posts is
  stored by Slack.
- **"Approve all <tool>"** lasts for the session. Use it sparingly.
- **Auto-memory notes persist.** Whatever is written to a project's memory affects later sessions
  there, including terminal sessions.
- **Terminal mirroring** sends the full content of mirrored terminal sessions to Slack, including
  tool outputs (protected-path outputs are omitted). Enable it only for directories where that is
  acceptable.
- **A bot restart ends the agent's own background commands.** Use `sac-job` for long work.

## Reporting a vulnerability

Please report security issues privately through GitHub's *Report a vulnerability* (Security
Advisories) on this repository, not in public issues. Include steps to reproduce. Expect an answer
within a week.
