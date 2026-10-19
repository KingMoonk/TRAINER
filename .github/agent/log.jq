# Turns Claude Code's stream-json output into a readable job log: what the agent said, which tools it called, failed tool
# results, and the final result line. The full stream is kept as transcript.jsonl in the run's agent-out artifact.
# Run with `jq -R -r`: each line is parsed here, so a line that is not JSON is skipped instead of failing the step.
def clip($n): tostring | gsub("\\s+"; " ") | if length > $n then .[0:$n] + "…" else . end;
def who: if .parent_tool_use_id then "  [sub] " else "" end;

fromjson? // empty |
if .type == "system" and .subtype == "init" then
  "== session \(.session_id) · model \(.model) · \(.tools | length) tools"
elif .type == "assistant" then
  who as $w | .message.content[]? |
  if .type == "text" then $w + "💬 " + (.text | clip(2000))
  elif .type == "tool_use" then
    $w + "🔧 " + .name + " " + ((.input.command // .input.file_path // .input.pattern // .input.url // .input.query // .input.description // .input) | clip(240))
  else empty end
elif .type == "user" then
  who as $w | .message.content[]? | select(type == "object" and .type == "tool_result" and .is_error == true) |
  $w + "   ↳ error: " + ((.content | if type == "array" then map(.text? // "") | join(" ") else . end) | clip(400))
elif .type == "system" and .subtype == "permission_denied" then
  "   ↳ denied: " + (. | clip(240))
elif .type == "result" then
  "== \(.subtype) · \(.num_turns) turns · $\((.total_cost_usd // 0) * 100 | round / 100) · \(((.duration_ms // 0) / 60000) | floor) min"
else empty end
