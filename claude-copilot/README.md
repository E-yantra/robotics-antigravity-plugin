# Robotics plugin for Claude Code and GitHub Copilot

Port of Google DeepMind's
[robotics-antigravity-plugin](https://github.com/google-deepmind/robotics-antigravity-plugin)
(Apache-2.0, version 0.1.12) to the Claude Code / Copilot plugin format. One
package serves both: Copilot CLI and VS Code agent plugins read the
`.claude-plugin/` manifest.

## Install

Claude Code:

```bash
claude plugin marketplace add E-yantra/robotics-antigravity-plugin   # once the claude-copilot branch is merged to the default branch
# or from a local clone (repo root, where .claude-plugin/marketplace.json lives):
claude plugin marketplace add /path/to/robotics-antigravity-plugin
claude plugin install robotics@robotics-marketplace
# or, for one session:  claude --plugin-dir /path/to/robotics-antigravity-plugin/claude-copilot
```

GitHub Copilot CLI:

```bash
copilot plugin install /path/to/robotics-antigravity-plugin/claude-copilot
```

The MuJoCo API skills (`mujoco-python`, `mujoco-studio`, ...) are separate:
`npx skills add google-deepmind/mujoco`. For the Gemini perception skills set
`GEMINI_API_KEY`.

## What changed from the Antigravity original

| Antigravity | Here |
| --- | --- |
| `plugins/robotics/plugin.json` | `.claude-plugin/plugin.json` (+ `marketplace.json`) |
| `rules/*.md` (`trigger: model_decision`) | Skills `mujoco-asset-standards`, `mujoco-studio-workflow` (same descriptions drive auto-loading) |
| Node sidecar + `sidecar://` aux-pane pill | Removed. `studiod.py` already serves the viewer; skills now use `--open_browser` and link `http://127.0.0.1:<http_port>/` |
| `invoke_subagent`, `browser` subagent, `run_command`, `read_file`, `search_web`, `IsDaemon`, `BypassSandbox` | Host-neutral wording (subagent, web search, file-read tool, background process) |
| `<appDataDir>/brain/...` transcript | Host transcript locations for Claude Code / Copilot (share-example) |

Skill scripts, references, and assets are unchanged. `studiod` still honors the
`ANTIGRAVITY_*` env vars as harmless fallbacks; the Node-sidecar notify path is a
no-op without a sidecar.

Untested on real hardware. See the upstream README for safety guidance on real robots.
