#!/bin/bash
# PreToolUse hook: block shell commands that read credential .env files.
#
# The permissions deny list only covers the Read tool, so `cat .env.private`
# (or grep/head/less/source/python) would otherwise slip past it. This hook
# closes that gap for Bash.
#
# Protected: .env, .env.private, .env.docker (credentials and build secrets).
# NOT protected: .env.shared (documented as non-secret configuration) and every
# *.example file (tracked in git, no real values).
#
# Scope: a protected file must appear as an argument to a known reading command.
# Matching the filename anywhere in the command string was tried first and is
# stricter, but it also blocks commands that merely *mention* the file -- a git
# commit message or a doc edit naming .env.private -- which made it unusable.
# The trade-off is that a reader outside the list below is not caught; this is
# defence-in-depth against an accidental read, not an airtight sandbox.
#
# Exit code 2 blocks the tool call; the stderr message is fed back to Claude.

command=$(python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null)

[[ -n "$command" ]] || exit 0

# Commands that would surface file contents (or load them into the environment).
readers='cat|bat|head|tail|less|more|grep|egrep|fgrep|rg|ag|ack|awk|sed|cut|sort|uniq|nl|tac|strings|xxd|od|hexdump|wc|diff|dotenv|source|\.|export|env|python|python3|ruby|perl|node|jq|tee|cp|mv|scp|rsync|curl'

# Drop quoted string literals, so prose that happens to contain a read -- most
# often a commit message like -m "blocks cat .env.private" -- is not mistaken
# for one. A genuine read written as `cat ".env.private"` is missed as a result;
# that is an accepted limit, since an accidental read is written unquoted.
# Text inside a heredoc body is not stripped and can still trigger a false
# positive; reword it or pass the message via -m.
unquoted=$(printf '%s' "$command" | sed -E "s/'[^']*'//g; s/\"[^\"]*\"//g")

# Drop "*.example" references next, so a template file is judged on its own and
# not on the protected name it embeds. Note this leaves the *destination* of
# `cp .env.private.example .env.private` visible, which is intended: that
# command overwrites real credentials and the user runs it by hand during setup.
stripped=$(printf '%s' "$unquoted" | sed -E 's/\.env[A-Za-z0-9._-]*\.example//g')

# A protected path as a standalone argument: optional ./ or absolute prefix,
# then .env / .env.private / .env.docker, ending at a word boundary so that
# .env.shared and .env.anything-else are not caught by the bare-.env rule.
protected='([^[:space:]]*/)?\.env(\.(private|docker))?([^.A-Za-z0-9_/-]|$)'

if printf '%s' "$stripped" | grep -qE "(^|[[:space:];&|(\`])($readers)([[:space:]]+[^[:space:]]+)*[[:space:]]+[\"']?$protected"; then
    echo "Blocked: this command reads .env, .env.private, or .env.docker, which hold credentials. Read the corresponding .example file instead, or ask the user for the specific value you need. (.env.shared is not restricted.)" >&2
    exit 2
fi

exit 0
