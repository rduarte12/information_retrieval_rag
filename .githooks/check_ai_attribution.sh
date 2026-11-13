#!/usr/bin/env bash
# Bloqueia atribuição de IA em commits (decisão D9).
#
# Uso:
#   check_ai_attribution.sh --msg-file <arquivo>   # usado pelo hook commit-msg
#   check_ai_attribution.sh --range <rev-range>    # usado pela GitHub Action
#
# Bloqueia:
#   - trailers Co-authored-by / Generated-by / Assisted-by / Signed-off-by que citem uma IA;
#   - linhas "Generated with <IA>", "Claude-Session:", o emoji 🤖;
#   - autor ou committer com e-mail de bot ou de provedor de IA.
set -euo pipefail

AI='claude|anthropic|openai|chatgpt|gpt-[0-9]|codex|copilot|cursor|gemini|bard|devin|aider|windsurf|codeium|tabnine|codewhisperer|amazon q|\[bot\]'
MSG_RE="^[[:space:]]*(co-authored-by|generated-by|assisted-by|signed-off-by)[[:space:]]*:.*(${AI})|generated (with|by) .*(${AI})|^[[:space:]]*claude-session:|🤖"
IDENT_RE="noreply@anthropic\.com|@anthropic\.com|@openai\.com|\[bot\]|copilot|cursoragent|cursor\.com|devin-ai"

fail=0

check_text() {  # $1 = rótulo, stdin = mensagem
  local hits
  hits=$(grep -v '^#' | grep -inE "$MSG_RE" || true)
  if [ -n "$hits" ]; then
    echo "✗ $1: mensagem com atribuição de IA:" >&2
    echo "$hits" | sed 's/^/    /' >&2
    fail=1
  fi
}

check_ident() {  # $1 = rótulo, $2 = "Nome <email>"
  if echo "$2" | grep -iqE "$IDENT_RE"; then
    echo "✗ $1: autor/committer não permitido: $2" >&2
    fail=1
  fi
}

case "${1:-}" in
  --msg-file)
    check_text "commit" < "$2"
    check_ident "autor" "$(git var GIT_AUTHOR_IDENT | sed -E 's/ [0-9]+ [-+][0-9]{4}$//')"
    check_ident "committer" "$(git var GIT_COMMITTER_IDENT | sed -E 's/ [0-9]+ [-+][0-9]{4}$//')"
    ;;
  --range)
    for c in $(git rev-list "$2"); do
      short=$(git rev-parse --short "$c")
      check_text "commit $short" < <(git log -1 --format=%B "$c")   # sem pipe: fail precisa ficar no shell atual
      check_ident "commit $short (autor)" "$(git log -1 --format='%an <%ae>' "$c")"
      check_ident "commit $short (committer)" "$(git log -1 --format='%cn <%ce>' "$c")"
    done
    ;;
  *)
    echo "uso: $0 --msg-file <arquivo> | --range <rev-range>" >&2
    exit 2
    ;;
esac

if [ "$fail" -ne 0 ]; then
  echo "" >&2
  echo "Commits deste repositório não podem ter IA como autora ou coautora (D9, AGENTS.md)." >&2
  echo "Remova as linhas indicadas e tente de novo." >&2
  exit 1
fi
