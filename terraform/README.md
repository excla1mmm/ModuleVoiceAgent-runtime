# Voice Agent runtime on Yandex Cloud

`shard_count` independent VMs (Ubuntu 24.04, Docker). On each shard Caddy terminates TLS with an automatic
Let's Encrypt certificate and proxies HTTPS and WSS to the runtime on the same host (`VA_TRUSTED_PROXY=true`).
The runtime keeps bundles, calls and results in RAM, so shards share nothing: **every PBX is pinned to one
shard** by its DNS name (`voice-1.example.com`, `voice-2.example.com`, ...) and listed only in that shard's
tenants file. There is no load balancer in front, by design.

```
PBX A, B ──wss/https──> voice-1.example.com : Caddy :443 ──> runtime :8443   (tenants/shard-1.json)
PBX C, D ──wss/https──> voice-2.example.com : Caddy :443 ──> runtime :8443   (tenants/shard-2.json)
```

What gets created: VPC network and subnet, a security group (22 from `admin_cidrs`, 80 from anywhere for
ACME HTTP-01, 443 from `allowed_cidrs`), a static IP per shard, A records in Cloud DNS (when `dns_zone_id`
is set), an optional Container Registry, a Lockbox secret with the LLM key(s) and the operator token, and a
VM service account allowed to pull images and read that secret. Secrets never travel in instance metadata:
on every boot the VM fetches them with its service-account token into `/etc/voice-agent/secrets`
(`LLM_API_KEY_FILE`, `VA_ADMIN_TOKEN_FILE`). They are still stored in the Terraform state; keep the state
in a protected backend.

Sizing: `VA_MAX_CALLS = 2 × cores` (4 vCPU = 8 concurrent calls), RAM ~1.5 GB + 60 MB per call.

## Deploy

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars && chmod 600 terraform.tfvars   # cloud, domain, keys
mkdir -p tenants   # tenants/shard-1.json, shard-2.json, ...: {"tenants": [...]} as in ../config
export YC_TOKEN=$(yc iam create-token)
terraform init
terraform apply -target=yandex_container_registry.this   # first run only: the registry for the image
scripts/build-push.sh latest                             # build ../Dockerfile and push
terraform apply
terraform output shards
curl -s https://voice-1.example.com/health
curl -s -H "Authorization: Bearer $(terraform output -raw admin_token)" https://voice-1.example.com/health
```

The first start downloads ~1.5 GB of models (set `hf_endpoint` to use a mirror) and obtains the
certificate; the DNS records must already point at the shard (`terraform output dns_records` when
`dns_zone_id` is empty). Progress: `ssh ubuntu@<ip> sudo docker compose --project-directory /etc/voice-agent logs -f`.

## Operations

* **Add a PBX**: put it into `tenants/shard-<N>.json` of the least loaded shard, run
  `scripts/sync-tenants.sh <N>` (atomic replace, the runtime re-reads the file, no restart), then set
  `https://voice-<N>.example.com` as the runtime URL on the PBX. The script refuses a PBX id listed on two
  shards.
* **Move a PBX**: add it to the new shard, sync, change the URL on the PBX, then remove it from the old
  shard and sync. Bundles are re-uploaded by the PBX heartbeat.
* **New image**: `DEPLOY=1 scripts/build-push.sh <tag>` pushes and switches shards one at a time (each
  drains: active calls get `VA_DRAIN_SECONDS`, new ones go to the PBX fallback until it is back), waiting
  for `/health` between shards. Set `runtime_image`/`image_tag` in `terraform.tfvars` too, for new shards.
* **More capacity**: raise `shard_count` (new shards, existing ones untouched) or `cores` (stops the VMs
  one by one for resize; update `VA_MAX_CALLS` in `/etc/voice-agent/.env` and run
  `sudo voice-agent-refresh`).
* **Rotate secrets**: change them in `terraform.tfvars`, `terraform apply`, then on each shard
  `sudo voice-agent-refresh && sudo docker compose --project-directory /etc/voice-agent restart runtime`.
* **Monitoring**: `/metrics` (Prometheus, series labelled by `tenant`) with the operator token.

cloud-init runs on the first boot only, and changes of `user-data` are ignored on existing shards (so a
new Ubuntu image or a template edit never recreates a running shard). To rebuild one shard from the
current template: `terraform apply -replace='yandex_compute_instance.shard[1]'` (its PBXs go to their
fallback number while it boots).

Docker Hub unreachable: set `docker_registry_mirrors`, a mirrored `caddy_image`, and `PYTHON_IMAGE` for
`scripts/build-push.sh`.
