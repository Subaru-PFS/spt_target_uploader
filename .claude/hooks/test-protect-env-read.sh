#!/bin/bash
# Test harness for .claude/hooks/protect-env-read.sh
HOOK="$1"
fail=0

check() {
    local want="$1" cmd="$2"
    local payload
    payload=$(python3 -c 'import json,sys;print(json.dumps({"tool_name":"Bash","tool_input":{"command":sys.argv[1]}}))' "$cmd")
    printf '%s' "$payload" | "$HOOK" >/dev/null 2>&1
    local got=$?
    if [[ "$got" == "$want" ]]; then
        printf '  ok   (%s) %s\n' "$got" "$cmd"
    else
        printf '  FAIL want=%s got=%s  %s\n' "$want" "$got" "$cmd"
        fail=1
    fi
}

echo "=== must BLOCK (exit 2) ==="
check 2 'cat .env.private'
check 2 'grep ADMIN_PASSWORD .env.private'
check 2 'head -20 .env.docker'
check 2 'cat .env'
check 2 'source .env.private'
check 2 'less ./.env.docker'
check 2 'cat /Users/monodera/Dropbox/NAOJ/PFS/spt_target_uploader/.env.private'
check 2 'sed -n 1,5p .env.private'
check 2 'git status && cat .env.private'
check 2 'cp .env.private /tmp/leak'
check 2 'tail -5 .env'
check 2 'python3 .env.private'

echo "=== must ALLOW (exit 0) ==="
check 0 'git commit -m "hook blocks cat .env.private and .env.docker"'
check 0 'echo "see .env.private for details"'
check 0 'cat .env.shared'
check 0 'cat .env.private.example'
check 0 'cat .env.shared.example'
check 0 'cp .env.shared.example .env.shared'
check 0 'git status'
check 0 'uv run pytest tests/'
check 0 'ls -a'
check 0 'grep -n LOG_LEVEL .env.shared'

exit $fail
