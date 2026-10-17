# Voice Agent runtime on Yandex Cloud: N independent shards.
#
# Every shard is one VM with Docker: Caddy terminates TLS (Let's Encrypt) and proxies to the runtime on the
# same host. Agent bundles, active calls and results live in the runtime's RAM, so shards share nothing and
# every PBX is pinned to one shard by its DNS name (voice-1.example.com, voice-2.example.com, ...).

locals {
  hostnames = [for i in range(var.shard_count) : "${format(var.hostname_format, i + 1)}.${var.domain}"]

  runtime_image = var.runtime_image != "" ? var.runtime_image : (
    var.create_registry ? "cr.yandex/${yandex_container_registry.this[0].id}/voice-agent-runtime:${var.image_tag}" : ""
  )

  admin_token = var.admin_token != "" ? var.admin_token : random_password.admin.result

  # Secrets reach the VM through Lockbox, never through instance metadata; the file names match the *_FILE settings.
  secrets = merge(
    { llm_api_key = var.llm_api_key, admin_token = local.admin_token },
    var.llm2_api_key != "" ? { llm2_api_key = var.llm2_api_key } : {},
  )

  env = merge(
    {
      VA_IMAGE            = local.runtime_image # read by compose (image:) and scripts/build-push.sh
      LLM_URL             = var.llm_url
      LLM_MODEL           = var.llm_model
      LLM_EXTRA_JSON      = var.llm_extra_json
      LLM_API_KEY_FILE    = "/run/secrets/llm_api_key"
      VA_ADMIN_TOKEN_FILE = "/run/secrets/admin_token"
      VA_TRUSTED_PROXY    = "true" # Caddy terminates TLS on the same host
      VA_PUBLIC_BIND      = "127.0.0.1"
      VA_LISTEN_PORT      = "8443"
      VA_TENANTS_FILE     = "/config/tenants.json"
      VA_MAX_CALLS        = tostring(2 * var.cores)
      VA_DRAIN_SECONDS    = "30"
      VA_USAGE_DB         = "/data/usage.sqlite3"
    },
    var.llm2_url != "" ? {
      LLM2_URL          = var.llm2_url
      LLM2_MODEL        = var.llm2_model
      LLM2_API_KEY_FILE = "/run/secrets/llm2_api_key"
    } : {},
    var.hf_endpoint != "" ? { HF_ENDPOINT = var.hf_endpoint } : {},
    var.extra_env,
  )
  env_file = join("\n", [for k in sort(keys(local.env)) : "${k}=${local.env[k]}"])

  daemon_json = jsonencode(length(var.docker_registry_mirrors) > 0 ? { registry-mirrors = var.docker_registry_mirrors } : {})
}

data "yandex_compute_image" "ubuntu" {
  family    = var.image_family
  folder_id = "standard-images"
}

resource "random_password" "admin" {
  length  = 40
  special = false
}

# --- Network ---

resource "yandex_vpc_network" "this" {
  name = var.name
}

resource "yandex_vpc_subnet" "this" {
  name           = "${var.name}-${var.zone}"
  zone           = var.zone
  network_id     = yandex_vpc_network.this.id
  v4_cidr_blocks = [var.subnet_cidr]
}

resource "yandex_vpc_security_group" "runtime" {
  name       = "${var.name}-runtime"
  network_id = yandex_vpc_network.this.id

  dynamic "ingress" {
    for_each = length(var.admin_cidrs) > 0 ? [1] : []
    content {
      description    = "SSH (operators)"
      protocol       = "TCP"
      port           = 22
      v4_cidr_blocks = var.admin_cidrs
    }
  }

  ingress {
    description    = "HTTP: Let's Encrypt HTTP-01 and redirect to HTTPS"
    protocol       = "TCP"
    port           = 80
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description    = "HTTPS / WSS from the PBXs"
    protocol       = "TCP"
    port           = 443
    v4_cidr_blocks = var.allowed_cidrs
  }

  egress {
    description    = "LLM providers, registries, Let's Encrypt, model download"
    protocol       = "ANY"
    from_port      = 0
    to_port        = 65535
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "yandex_vpc_address" "shard" {
  count               = var.shard_count
  name                = "${var.name}-${count.index + 1}"
  deletion_protection = var.deletion_protection

  external_ipv4_address {
    zone_id = var.zone
  }
}

resource "yandex_dns_recordset" "shard" {
  count   = var.dns_zone_id != "" ? var.shard_count : 0
  zone_id = var.dns_zone_id
  name    = "${local.hostnames[count.index]}."
  type    = "A"
  ttl     = 300
  data    = [yandex_vpc_address.shard[count.index].external_ipv4_address[0].address]
}

# --- Image registry and secrets ---

resource "yandex_container_registry" "this" {
  count = var.create_registry ? 1 : 0
  name  = var.name
}

resource "yandex_iam_service_account" "vm" {
  name        = "${var.name}-vm"
  description = "Voice Agent runtime VMs: pull the image, read the Lockbox secret"
}

resource "yandex_resourcemanager_folder_iam_member" "vm_puller" {
  folder_id = var.folder_id
  role      = "container-registry.images.puller"
  member    = "serviceAccount:${yandex_iam_service_account.vm.id}"
}

resource "yandex_lockbox_secret" "runtime" {
  name                = "${var.name}-runtime"
  description         = "LLM keys and the operator token of the Voice Agent runtime"
  deletion_protection = var.deletion_protection
}

resource "yandex_lockbox_secret_version" "runtime" {
  secret_id = yandex_lockbox_secret.runtime.id

  dynamic "entries" {
    for_each = local.secrets
    content {
      key        = entries.key
      text_value = entries.value
    }
  }
}

resource "yandex_lockbox_secret_iam_binding" "vm" {
  secret_id = yandex_lockbox_secret.runtime.id
  role      = "lockbox.payloadViewer"
  members   = ["serviceAccount:${yandex_iam_service_account.vm.id}"]
}

# --- Shards ---

resource "yandex_compute_instance" "shard" {
  count                     = var.shard_count
  name                      = "${var.name}-${count.index + 1}"
  hostname                  = "${var.name}-${count.index + 1}"
  platform_id               = var.platform_id
  zone                      = var.zone
  service_account_id        = yandex_iam_service_account.vm.id
  allow_stopping_for_update = true

  labels = {
    app   = "voice-agent-runtime"
    shard = tostring(count.index + 1)
  }

  resources {
    cores         = var.cores
    memory        = var.memory
    core_fraction = var.core_fraction
  }

  boot_disk {
    initialize_params {
      image_id = data.yandex_compute_image.ubuntu.id
      size     = var.disk_size
      type     = var.disk_type
    }
  }

  network_interface {
    subnet_id          = yandex_vpc_subnet.this.id
    nat                = true
    nat_ip_address     = yandex_vpc_address.shard[count.index].external_ipv4_address[0].address
    security_group_ids = [yandex_vpc_security_group.runtime.id]
  }

  scheduling_policy {
    preemptible = var.preemptible
  }

  metadata = {
    user-data = templatefile("${path.module}/cloud-init.yaml.tftpl", {
      ssh_user          = var.ssh_user
      ssh_public_key    = var.ssh_public_key
      hostname          = local.hostnames[count.index]
      acme_email        = var.acme_email
      caddy_image       = var.caddy_image
      lockbox_secret_id = yandex_lockbox_secret.runtime.id
      env_file          = local.env_file
      daemon_json       = local.daemon_json
      tenants_json = fileexists("${path.module}/${var.tenants_dir}/shard-${count.index + 1}.json") ? file(
        "${path.module}/${var.tenants_dir}/shard-${count.index + 1}.json"
      ) : jsonencode({ tenants = [] })
    })
  }

  lifecycle {
    # cloud-init runs on the first boot only. Later changes go through scripts/ (image, tenants) or
    # `terraform apply -replace=...` of a shard; a new Ubuntu image must not recreate running shards.
    ignore_changes = [metadata["user-data"], boot_disk[0].initialize_params[0].image_id]

    precondition {
      condition     = local.runtime_image != ""
      error_message = "Set runtime_image or create_registry = true."
    }
  }

  depends_on = [
    yandex_resourcemanager_folder_iam_member.vm_puller,
    yandex_lockbox_secret_iam_binding.vm,
    yandex_lockbox_secret_version.runtime,
  ]
}
