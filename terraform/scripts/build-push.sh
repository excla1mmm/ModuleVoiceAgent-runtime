#!/usr/bin/env bash
# Builds the runtime image, pushes it to Container Registry and optionally rolls it out shard by shard.
#
#   scripts/build-push.sh [tag]            build and push (tag defaults to a UTC timestamp)
#   DEPLOY=1 scripts/build-push.sh [tag]   also switch every shard to the new image, one at a time
#
# Env: REGISTRY (default: terraform output registry), PLATFORM (linux/amd64),
#      PYTHON_IMAGE (base image, e.g. a Docker Hub mirror), SSH_OPTS.
# On a shard the switch drains it: new calls go to the PBX fallback for a few seconds, active calls get
# VA_DRAIN_SECONDS to finish. Set runtime_image in terraform.tfvars too, so new shards start with it.
set -euo pipefail

tf_dir=$(cd "$(dirname "$0")/.." && pwd)
repo_dir=$(cd "$tf_dir/.." && pwd)
cd "$tf_dir"

registry=${REGISTRY:-$(terraform output -raw registry 2>/dev/null || true)}
[ -n "$registry" ] || { echo "REGISTRY is not set and terraform output registry is empty" >&2; exit 1; }
tag=${1:-$(date -u +%Y%m%d%H%M%S)}
image="$registry/voice-agent-runtime:$tag"

build_args=()
[ -n "${PYTHON_IMAGE:-}" ] && build_args+=(--build-arg "PYTHON_IMAGE=$PYTHON_IMAGE")

yc container registry configure-docker >/dev/null
docker buildx build --platform "${PLATFORM:-linux/amd64}" ${build_args[@]+"${build_args[@]}"} -t "$image" --push "$repo_dir"
echo "pushed $image"

[ "${DEPLOY:-}" = 1 ] || exit 0

user=$(terraform output -raw ssh_user)
terraform output -json shards | jq -r '.[] | "\(.ip) \(.url)"' | while read -r ip url; do
  echo "== $url ($ip)"
  # shellcheck disable=SC2086
  ssh -n ${SSH_OPTS:-} "$user@$ip" \
    "sudo sed -i 's|^VA_IMAGE=.*|VA_IMAGE=$image|' /etc/voice-agent/.env && sudo /usr/local/bin/voice-agent-refresh"
  for _ in $(seq 60); do
    curl -sf --max-time 5 "$url/health" >/dev/null && break
    sleep 5
  done
  curl -sf --max-time 5 "$url/health" >/dev/null || { echo "$url is not healthy, stopping the rollout" >&2; exit 1; }
done
