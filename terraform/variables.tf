# --- Cloud ---

variable "cloud_id" {
  type = string
}

variable "folder_id" {
  type = string
}

variable "zone" {
  type    = string
  default = "ru-central1-a"
}

variable "service_account_key_file" {
  description = "Authorized key of the deploying service account; empty = YC_TOKEN from the environment."
  type        = string
  default     = ""
}

variable "name" {
  description = "Prefix of every resource name."
  type        = string
  default     = "voice-agent"
}

# --- Shards ---

variable "shard_count" {
  description = "Independent runtime VMs. State lives in RAM, so every PBX is pinned to one shard by DNS name."
  type        = number
  default     = 1

  validation {
    condition     = var.shard_count >= 1 && var.shard_count <= 50
    error_message = "shard_count must be between 1 and 50."
  }
}

variable "domain" {
  description = "DNS domain of the shards, e.g. example.com."
  type        = string
}

variable "hostname_format" {
  description = "printf format of a shard host name inside var.domain; %d is the 1-based shard number."
  type        = string
  default     = "voice-%d"
}

variable "dns_zone_id" {
  description = "Cloud DNS public zone that serves var.domain; empty = create the A records yourself (see outputs)."
  type        = string
  default     = ""
}

variable "platform_id" {
  type    = string
  default = "standard-v3"
}

variable "cores" {
  description = "vCPUs per shard; the runtime takes 2 concurrent calls per vCPU (VA_MAX_CALLS)."
  type        = number
  default     = 4

  validation {
    condition     = var.cores >= 2
    error_message = "cores must be at least 2."
  }
}

variable "memory" {
  description = "RAM per shard, GB: ~1.5 GB + 60 MB per call, plus Caddy and the OS."
  type        = number
  default     = 4
}

variable "core_fraction" {
  description = "Guaranteed vCPU share. Keep 100: speech recognition is CPU-bound and latency-sensitive."
  type        = number
  default     = 100
}

variable "disk_size" {
  description = "Boot disk, GB (OS, Docker images, ~1.5 GB of models)."
  type        = number
  default     = 30
}

variable "disk_type" {
  type    = string
  default = "network-ssd"
}

variable "image_family" {
  type    = string
  default = "ubuntu-2404-lts"
}

variable "preemptible" {
  description = "Preemptible VMs are stopped at least once a day; lab use only."
  type        = bool
  default     = false
}

variable "deletion_protection" {
  description = "Protect the static IPs (PBXs and DNS point at them)."
  type        = bool
  default     = true
}

# --- Network ---

variable "subnet_cidr" {
  type    = string
  default = "10.130.0.0/24"
}

variable "admin_cidrs" {
  description = "Networks allowed to SSH into the shards; empty = no SSH."
  type        = list(string)
  default     = []
}

variable "allowed_cidrs" {
  description = "Networks allowed to reach 443 (the PBXs). Port 80 stays open for Let's Encrypt HTTP-01."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "ssh_user" {
  type    = string
  default = "ubuntu"
}

variable "ssh_public_key" {
  type = string
}

# --- Images ---

variable "runtime_image" {
  description = "Full runtime image reference; empty = cr.yandex/<created registry>/voice-agent-runtime:<image_tag>."
  type        = string
  default     = ""
}

variable "create_registry" {
  description = "Create a Container Registry for the runtime image (scripts/build-push.sh pushes there)."
  type        = bool
  default     = true
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "caddy_image" {
  description = "TLS proxy image; point it at a mirror when Docker Hub is unreachable."
  type        = string
  default     = "caddy:2.8"
}

variable "docker_registry_mirrors" {
  description = "Docker Hub mirrors for the shards (daemon.json registry-mirrors), e.g. [\"https://mirror.gcr.io\"]."
  type        = list(string)
  default     = []
}

variable "hf_endpoint" {
  description = "HF_ENDPOINT for the model download on first start (Hugging Face mirror); empty = huggingface.co."
  type        = string
  default     = ""
}

# --- Runtime settings ---

variable "acme_email" {
  description = "Let's Encrypt account e-mail (expiry notices); may be empty."
  type        = string
  default     = ""
}

variable "llm_url" {
  type    = string
  default = "https://foundation-models.api.cloud.ru/v1/chat/completions"
}

variable "llm_model" {
  type    = string
  default = "openai/gpt-oss-120b"
}

variable "llm_extra_json" {
  type    = string
  default = "{\"reasoning_effort\":\"low\"}"
}

variable "llm_api_key" {
  description = "Primary LLM key; stored in Lockbox (and in the Terraform state)."
  type        = string
  sensitive   = true
}

variable "llm2_url" {
  description = "Optional secondary LLM provider."
  type        = string
  default     = ""
}

variable "llm2_model" {
  type    = string
  default = ""
}

variable "llm2_api_key" {
  type      = string
  default   = ""
  sensitive = true
}

variable "admin_token" {
  description = "Operator token for /metrics and the full /health; empty = generated."
  type        = string
  default     = ""
  sensitive   = true
}

variable "extra_env" {
  description = "Additional runtime environment (VA_MAX_CALLS_PER_TENANT, VA_LOG_LEVEL, ...); overrides the defaults."
  type        = map(string)
  default     = {}
}

variable "tenants_dir" {
  description = "Directory with shard-<N>.json tenant files, written on first boot and by scripts/sync-tenants.sh."
  type        = string
  default     = "tenants"
}
