output "shards" {
  description = "One entry per shard: point each PBX at exactly one url (and list it in that shard's tenants file)."
  value = [for i, vm in yandex_compute_instance.shard : {
    number   = i + 1
    name     = vm.name
    hostname = local.hostnames[i]
    ip       = yandex_vpc_address.shard[i].external_ipv4_address[0].address
    url      = "https://${local.hostnames[i]}"
    tenants  = "${var.tenants_dir}/shard-${i + 1}.json"
  }]
}

output "dns_records" {
  description = "A records to create when dns_zone_id is empty."
  value       = { for i, h in local.hostnames : h => yandex_vpc_address.shard[i].external_ipv4_address[0].address }
}

output "ssh_user" {
  value = var.ssh_user
}

output "runtime_image" {
  value = local.runtime_image
}

output "registry" {
  description = "Push target for scripts/build-push.sh (empty without create_registry)."
  value       = var.create_registry ? "cr.yandex/${yandex_container_registry.this[0].id}" : ""
}

output "lockbox_secret_id" {
  value = yandex_lockbox_secret.runtime.id
}

output "admin_token" {
  description = "Operator Bearer token for /metrics and the full /health: terraform output -raw admin_token."
  value       = local.admin_token
  sensitive   = true
}
