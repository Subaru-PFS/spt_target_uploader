#!/bin/bash
# PreToolUse hook: block Claude Code from editing credential files and generated files.
# Exit code 2 blocks the tool call; the stderr message is fed back to Claude.

file_path=$(python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)

[[ -n "$file_path" ]] || exit 0

basename=$(basename "$file_path")

case "$basename" in
    .env|.env.shared|.env.private|.env.docker)
        echo "Blocked: $basename contains local/private configuration. Edit the corresponding .example file instead, or ask the user to update it manually." >&2
        exit 2
        ;;
    uv.lock|pdm.lock)
        echo "Blocked: $basename is a generated lock file. Run 'uv sync' / 'pdm install' to regenerate it instead of editing by hand." >&2
        exit 2
        ;;
    requirements.txt)
        echo "Blocked: requirements.txt is generated. Run ./scripts/gen-requirements.sh instead of editing by hand." >&2
        exit 2
        ;;
esac

exit 0
