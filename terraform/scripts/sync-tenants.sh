#!/usr/bin/env bash
# Copies tenants/shard-<N>.json to shard N. The runtime re-reads the file on change; no restart.
#
#   scripts/sync-tenants.sh          all shards
#   scripts/sync-tenants.sh 2 3      only shards 2 and 3
#
# A PBX must be listed on exactly one shard (the one its runtime URL points at). Env: SSH_OPTS.
set -euo pipefail

cd "$(dirname "$0")/.."
tenants_dir=tenants
user=$(terraform output -raw ssh_user)
shards=$(terraform output -json shards)

files=("$tenants_dir"/shard-*.json)
if [ -e "${files[0]}" ]; then
  for f in "${files[@]}"; do
    jq -e '.tenants | type == "array"' "$f" >/dev/null || { echo "$f: expected {\"tenants\": [...]}" >&2; exit 1; }
  done
  dupes=$(jq -r '.tenants[].id' "${files[@]}" | sort | uniq -d)
  [ -z "$dupes" ] || { echo "PBX ids listed on more than one shard: $dupes" >&2; exit 1; }
fi

wanted=" $* "
echo "$shards" | jq -r '.[] | "\(.number) \(.ip)"' | while read -r n ip; do
  [ $# -eq 0 ] || [[ "$wanted" == *" $n "* ]] || continue
  f="$tenants_dir/shard-$n.json"
  [ -f "$f" ] || { echo "shard $n: no $f, skipped"; continue; }
  # shellcheck disable=SC2086
  scp -q ${SSH_OPTS:-} "$f" "$user@$ip:/tmp/tenants.json"
  # Atomic replace inside the mounted directory, so the runtime never reads a half-written file.
  # shellcheck disable=SC2086
  ssh -n ${SSH_OPTS:-} "$user@$ip" 'sudo install -m 0644 /tmp/tenants.json /etc/voice-agent/config/.tenants.json.new \
    && sudo mv -f /etc/voice-agent/config/.tenants.json.new /etc/voice-agent/config/tenants.json && rm -f /tmp/tenants.json'
  echo "shard $n ($ip): $(jq '.tenants | length' "$f") PBXs"
done
