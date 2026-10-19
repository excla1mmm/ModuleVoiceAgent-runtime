#!/usr/bin/env bash
# Blue-green update of the runtime (compose.bluegreen.yaml) without dropping calls.
#
#   scripts/deploy.sh [--build] [--image IMAGE] [--restart] [--force]
#
# 1. Starts the color that is not running (blue <-> green) with the new image, next to the running one.
# 2. Waits until it is ready: models loaded, the agent bundles taken over from the running instance (memory to
#    memory), greetings synthesized. Caddy routes new calls to it from then on.
# 3. Drains the old instance: no new calls, active calls end normally, the PBXs collect their results (asked
#    at the new instance, which forwards to the old one). Then the old instance is stopped.
# If the new instance does not become ready, it is stopped and the old one keeps serving.
#
# Two runtimes need RAM for two: the script refuses below VA_DEPLOY_MIN_FREE_MB of MemAvailable. --restart is
# the fallback for a small host: it waits for a moment without calls and without results the PBXs have not
# collected yet (up to VA_DEPLOY_IDLE_WAIT seconds), saves its bundles to the data volume, stops it and starts
# the new one, which loads them (and deletes the file) before it is ready; calls during the start (about a
# minute) go to the PBX fallback numbers. Waiting for the results matters: a draining
# instance is no longer "ready", the proxy stops sending it requests, and with no second instance a PBX that
# collects the result of a call that has just ended gets an error and the result is lost.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${VA_IMAGE:-voice-agent-runtime:latest}"
MIN_FREE_MB="${VA_DEPLOY_MIN_FREE_MB:-2300}"
READY_TIMEOUT="${VA_DEPLOY_READY_TIMEOUT:-900}"
DRAIN_WAIT="${VA_DEPLOY_DRAIN_WAIT:-1860}"
IDLE_WAIT="${VA_DEPLOY_IDLE_WAIT:-600}"
BUILD=0 RESTART=0 FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) BUILD=1 ;;
    --image) IMAGE="$2"; shift ;;
    --restart) RESTART=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

compose() { VA_IMAGE="$IMAGE" docker compose -f compose.bluegreen.yaml "$@"; }
log() { echo "[$(date '+%H:%M:%S')] $*"; }
running() { [[ "$(docker inspect -f '{{.State.Running}}' "voice-agent-runtime-$1" 2>/dev/null)" == true ]]; }
in_runtime() { local color="$1"; shift; docker exec "voice-agent-runtime-$color" voice-agent-runtime "$@"; }
field() { sed -n "s/.*\"$1\": \\([^,}]*\\).*/\\1/p"; }

grep -qE '^VA_ADMIN_TOKEN(_FILE)?=.+' .env || {
  echo 'VA_ADMIN_TOKEN is required in .env: echo "VA_ADMIN_TOKEN=$(openssl rand -hex 32)" >> .env' >&2; exit 2; }

old=""
for color in blue green; do
  if running "$color"; then
    [[ -z "$old" ]] || { echo "both colors are running: finish or stop the previous deploy first" >&2; exit 1; }
    old="$color"
  fi
done
new=$([[ "$old" == blue ]] && echo green || echo blue)

if [[ "$BUILD" == 1 ]]; then
  log "building $IMAGE"
  docker build -t "$IMAGE" .
fi
if [[ -n "$old" ]]; then
  docker tag "$(docker inspect -f '{{.Image}}' "voice-agent-runtime-$old")" voice-agent-runtime:previous
  log "running: $old ($(in_runtime "$old" status || echo '?')); rollback image: voice-agent-runtime:previous"
fi

compose up -d proxy

wait_ready() {
  local deadline=$((SECONDS + READY_TIMEOUT))
  until in_runtime "$new" healthcheck --ready >/dev/null 2>&1; do
    if ! running "$new"; then log "$new exited:"; docker logs --tail 30 "voice-agent-runtime-$new" >&2; return 1; fi
    if (( SECONDS >= deadline )); then log "$new is not ready after ${READY_TIMEOUT}s"; return 1; fi
    sleep 3
  done
}

if [[ "$RESTART" == 1 || -z "$old" ]]; then
  if [[ -n "$old" ]]; then
    log "waiting up to ${IDLE_WAIT}s for a moment without calls and uncollected results on $old"
    deadline=$((SECONDS + IDLE_WAIT))
    idle() { local s; s="$(in_runtime "$old" status)"; [[ "$(field active_calls <<<"$s")" == 0 && "$(field pending_results <<<"$s")" == 0 ]]; }
    until idle; do
      if (( SECONDS >= deadline )); then
        [[ "$FORCE" == 1 ]] || { echo "calls never stopped; --force stops them (they go to the fallback numbers)" >&2; exit 1; }
        break
      fi
      sleep 2
    done
    in_runtime "$old" drain --wait 0 --settle 20 || true
    # the new instance loads them before it is ready: calls right after the start find their agents
    in_runtime "$old" save-bundles || log "bundles not saved: the PBXs push them again within ~10 s"
    compose stop "runtime-$old"
  fi
  log "starting $new"
  compose up -d --no-deps --force-recreate "runtime-$new"
  wait_ready || { compose stop "runtime-$new"; exit 1; }
  log "$new is ready"
  exit 0
fi

avail=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
if (( avail < MIN_FREE_MB )) && [[ "$FORCE" != 1 ]]; then
  echo "only ${avail} MB of RAM available, a second runtime needs ${MIN_FREE_MB} MB (VA_DEPLOY_MIN_FREE_MB):" >&2
  echo "add RAM, or update with --restart (a short pause in service at a moment without calls)" >&2
  exit 1
fi

log "starting $new next to $old"
compose up -d --no-deps --force-recreate "runtime-$new"
if ! wait_ready; then
  log "keeping $old, stopping $new"
  compose stop "runtime-$new"
  exit 1
fi
log "$new is ready ($(in_runtime "$new" status)); draining $old"
sleep 3  # the proxy checks /ready every second
set +e
in_runtime "$old" drain --wait "$DRAIN_WAIT"
rc=$?
set -e
[[ $rc == 0 ]] || log "drain of $old ended with code $rc: stopping it hangs up what is left (fallback numbers)"
compose stop "runtime-$old"
log "done: $new serves, $old stopped"
