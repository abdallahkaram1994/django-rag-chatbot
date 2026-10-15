#!/usr/bin/env bash
# Usage: scripts/wait_for_deploy.sh <base-url> <commit-sha> [timeout-seconds]
# Polls <base-url>/version until it reports <commit-sha>. Exit 0 when live, 1 on timeout.
set -u
url="${1%/}"
sha="$2"
timeout="${3:-1200}"
interval="${POLL_INTERVAL:-15}"
elapsed=0

while [ "$elapsed" -lt "$timeout" ]; do
  live="$(curl -fsS --max-time 60 "$url/version" 2>/dev/null | jq -r '.commit // empty' 2>/dev/null || true)"
  if [ "$live" = "$sha" ]; then
    echo "Live: $url is running $sha"
    exit 0
  fi
  echo "[$elapsed s] waiting... currently serving: ${live:-no response}"
  sleep "$interval"
  elapsed=$((elapsed + interval))
done

echo "Timed out after ${timeout}s waiting for $sha at $url" >&2
exit 1
