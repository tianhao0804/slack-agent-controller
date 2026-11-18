# slack-agent-controller

在你自己的 Linux 服务器上运行一个 Claude 智能体，用 Slack 来驱动它：
- 看它工作，用按钮批准它的操作；
- 会话可以在 Slack 和终端之间来回交接；
- 长时间的计算跑完后会通知你。

这些都可以在手机上完成。

[English README](README.md)

> slack-agent-controller 是一个独立的开源项目，通过 [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview) *powered by Claude*，**不是** Anthropic 的产品，也没有得到 Anthropic 的背书。使用 Agent SDK 需要遵守 [Anthropic 商业条款](https://www.anthropic.com/legal/commercial-terms)，你需要自己的 Anthropic API key 或 Amazon Bedrock 访问权限。

**部署前请先读 [SECURITY.md](SECURITY.md)。** 这个 bot 让 Slack 工作区里的人可以在你的机器上执行代码。各种防护能降低风险，但不能消除风险。

## 功能

- **一个 Slack thread 就是一个智能体会话**：在 thread 里回复就是接着聊。会话保存在磁盘上，重启后自动接续。回复会实时流式显示，每个工具调用都看得到。
- **按钮审批**：
  - 执行命令、改文件、访问网络时弹出卡片，有 Approve、Deny、本会话内全部批准这几个按钮。可以安全重复的命令还会多一个「Run & remember here」。
  - 30 分钟没人回应，自动拒绝。
  - 每个 thread 可以选两种模式：`ask` 模式下，所有有副作用的操作都要你点；`auto` 模式下，常规操作交给 Claude Code 的 auto-mode 分类器判断，但受保护文件、凭据、导出环境变量、往外传数据仍然会弹到 Slack。
- **审批也不能越过的硬限制**：
  - `~/.ssh`、`~/.aws`、`.env`、`*.pem`、bot 自己的文件等密钥和配置路径，一律禁止访问。用通配符或花括号展开间接指向它们，同样会被拦下。
  - 引用凭据的命令会被拒绝。
  - 只能在你允许的目录里工作。
- **适合手机使用**：
  - 每条回复下面有按钮：停止、交给终端、切换模式、查看状态；
  - 智能体正忙时你又发了消息，可以选择「中断并立刻发送」；
  - Claude 的澄清问题显示成可以点选的选项；
  - 你附上的截图、日志、PDF 会交给 Claude 看；
  - Slack 的 Home 标签是总览：需要你处理的事、正在运行的会话、作业和 GPU。
- **长计算作业（`sac-job`）**：训练、模拟这类计算作为独立的 systemd 用户单元运行。
  - 自动挑空闲的 GPU，没有空卡时排队；
  - 作业的环境里去掉了凭据；
  - 进度、报错和结果会发回 thread，并带「取消」按钮。
  - 智能体或 bot 重启都不影响作业。
- **终端和 Slack 配合（可选）**：
  - 终端里的会话可以镜像到 Slack，每一轮发一条消息，并附上完整的 Markdown 日志；
  - 终端开着这个会话时，Slack 这边会等着，终端退出后就能在 Slack 接着聊；
  - 在 Slack 里发 `!exit`，可以把会话交还给终端。
- **其他**：
  - 审计日志记录每一次工具调用的决定；
  - 单轮费用上限，以及每个会话的费用估算；
  - 被重启打断的消息会如实报告，不会自动重发；
  - 长时间没动静会提醒你；
  - 项目快捷方式；
  - 用 `!import` 导入已有会话、用 `!fork` 分叉会话；
  - 每个 thread 可以单独设置模型和 effort。

## 环境要求

| | |
|---|---|
| 操作系统 | **Linux** 上所有功能都能用，核心 bot 也能在 macOS 上运行。终端镜像需要 Linux；`sac-job` 需要 systemd 用户服务（运行 `loginctl enable-linger $USER`，这样你退出登录后服务也不会停）。 |
| Python | 3.11 或更新版本 |
| 模型 | **Amazon Bedrock**（API key、profile 或 access key）或 **Anthropic API key**。不支持 claude.ai 登录，Anthropic 不允许第三方工具使用这种登录方式。auto 模式需要支持它的模型。 |
| Slack | 一个你有权限创建并安装 App 的工作区 |
| 可选 | `nvidia-smi`（显示 GPU 状态、运行 GPU 作业）、`squeue`（Slurm 状态） |

## 快速开始

1. **创建 Slack App**：
   - 在 api.slack.com/apps 选择 *Create New App → From a manifest*，粘贴 [`slack_manifest.yaml`](slack_manifest.yaml)，然后安装到工作区；
   - 记下 *Bot User OAuth Token*（`xoxb-…`）；
   - 在 *Basic Information → App-Level Tokens* 里创建一个 scope 为 `connections:write` 的 token（`xapp-…`）；
   - 记下你的成员 ID（个人资料 → ⋯ → *Copy member ID*）。
2. **安装**：
   ```bash
   git clone https://github.com/tianhao0804/slack-agent-controller ~/slack-agent-controller
   cd ~/slack-agent-controller
   python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```
3. **配置**：
   ```bash
   mkdir -p ~/.config/slack-agent-controller
   cp .env.example ~/.config/slack-agent-controller/env
   chmod 600 ~/.config/slack-agent-controller/env
   $EDITOR ~/.config/slack-agent-controller/env   # 填 token、你的成员 ID、PROVIDER、CWD_ROOTS
   .venv/bin/python bot.py --check
   ```
4. **运行**：先运行 `.venv/bin/python bot.py`，私信 bot 试一下。要长期运行的话：
   ```bash
   mkdir -p ~/.config/systemd/user
   cp deploy/slack-agent-controller.service ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now slack-agent-controller
   loginctl enable-linger "$USER"
   ```
   单元文件默认仓库放在 `~/slack-agent-controller`，放在别处请修改里面的路径。
5. **可选功能**：
   - **终端镜像**：设好 `MIRROR_DIRS`、`MIRROR_CHANNEL`、`MIRROR_USER`，然后运行 `.venv/bin/python deploy/install_hooks.py`。它会往 `~/.claude/settings.json` 里加 hook，修改前会先备份，`--remove` 可以卸载。
   - **在终端里提交作业**：`ln -s ~/slack-agent-controller/deploy/sac-job ~/bin/sac-job`。

## 使用

私信 bot，或者在 `ALLOWED_CHANNEL_IDS` 里列出的私有频道里发消息。第一条消息开始一个会话，在它的 thread 里回复就是接着聊。

| 命令 | 作用 |
|---|---|
| `!new <目录\|别名> [消息]` | 在某个目录（必须在 `CWD_ROOTS` 之下）或 `PROJECTS` 别名对应的目录里开新会话 |
| `!projects` | 列出项目快捷方式，每个带「New session」按钮 |
| `!sessions [all]` | 会话列表，带标题、状态和「Open」按钮（`all` 会附上最近的终端会话） |
| `!stop` · `!stop bg` · `!stop all` | 停止当前回复并清空排队的消息 · 停止后台任务 · 两者都停 |
| `!status` | GPU、Slurm 队列、本会话的 ID、token 数、费用、模型和模式 |
| `!mode [ask\|auto]` | 本 thread 的权限模式 |
| `!model [...]` · `!effort [...]` | 本 thread 的模型和 effort（不带参数时显示当前设置和上次实际用的） |
| `!allowlist [remove <id>\|clear]` | 查看或撤销用「Run & remember here」记住的命令 |
| `!jobs` | 用 `sac-job` 提交的作业 |
| `!upload <路径>` | 把会话目录里的文件发到 Slack |
| `!resume <id>` · `!import <id…>` · `!fork` | 接入一个终端会话 · 导入会话并带上最近几轮对话 · 分叉当前会话 |
| `!exit` | 关掉本 thread 的智能体进程，把会话交给终端 |
| `!end` | 结束会话并解除和 thread 的绑定（会话记录保留） |
| `!unlock` | 解除残留的终端锁 |
| `!help` | 全部命令 |

**文件**：在消息里附上截图、日志、PDF 或 notebook，带不带文字都可以。文件会下载到 `~/.cache/slack-agent-controller/inbox/`，不放进你的项目目录，7 天后自动删除。Claude 用 Read 工具读取。

**作业**：直接对智能体说"用 2 张 GPU 把 X 作为作业跑"，或者自己运行：

```bash
sac-job submit --name relax --gpus 2 -- python relax.py --cfg run.yaml
sac-job list | status <id> | log <id> -n 100 | cancel <id>
```

**记忆**：Claude Code 的自动记忆是打开的，但每个会话只能读写**它自己项目**的记忆目录 `~/.claude/projects/<项目>/memory/`，和你在终端同一目录下用的是同一份。`~/.claude` 里的其他内容仍然禁止访问。

全部配置项见 [`.env.example`](.env.example)，每一项都有注释。

## 文件与日志

| 路径 | 内容 |
|---|---|
| `~/.config/slack-agent-controller/env` | 配置和密钥（权限 600） |
| `~/.local/state/slack-agent-controller/` | `sessions.json`（thread 和会话的对应关系）、`audit.jsonl`（每次工具调用的决定，会轮转）、`bot.log`、`allowlist.json`、`tasks.json`、`mirror.json`。智能体读不到这个目录 |
| `~/.local/share/slack-agent-controller/jobs/<id>/` | `sac-job` 的作业记录和日志 |
| `~/.cache/slack-agent-controller/inbox/` | 你在 Slack 里附上的文件 |
| `journalctl --user -u slack-agent-controller -f` | 服务日志 |

## 常见问题

| 现象 | 检查 |
|---|---|
| `config error: … chmod 600` | 运行 `chmod 600 ~/.config/slack-agent-controller/env` |
| 私信 bot 没反应 | 你的成员 ID 是否在 `ALLOWED_SLACK_USER_IDS` 里；manifest 里是否有 `messages_tab_read_only_enabled: false`；看一下 `journalctl --user -u slack-agent-controller` |
| 点按钮没反应 | App 设置里是否打开了 Interactivity |
| 提示缺少 `files:read` 权限 | 在 *OAuth & Permissions* 里加上这个权限，然后重新安装 App |
| Home 标签是空的 | 打开 *Home Tab*，加上 `app_home_opened` 事件，然后重新安装 App |
| 提示 "another slack-agent-controller is already running" | 同一台机器上同一个用户只能运行一个实例，先停掉另一个 |
| 认证失败 | 模型提供方的 key 过期了：在 env 文件里更新，然后重启服务 |

## 开发

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/ruff check slack_agent_controller tests deploy bot.py
.venv/bin/mypy --ignore-missing-imports slack_agent_controller bot.py
.venv/bin/pytest -q          # 离线测试：假 Slack，不调用模型
```

集成测试（`tests/integration_*.py`）会真实调用模型，Slack 是假的。需要逐个运行，比如 `python -m tests.integration_fake_slack`。运行时环境里要有模型提供方的凭据，每个大约花几美分。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

**兼容性**：已测试的版本是 `claude-agent-sdk` 0.2.182，它自带 Claude Code 2.1.325。有几个功能依赖 Claude Code 的内部细节，可能随版本变化：会话记录的格式、hook 的字段、后台会话的进程结构、`autoMemoryDirectory` 设置。建议固定 SDK 版本，升级前先把集成测试跑一遍。

## 许可证

[MIT](LICENSE)。
