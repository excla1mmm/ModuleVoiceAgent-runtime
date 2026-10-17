terraform {
  required_version = ">= 1.5"

  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.120"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.6"
    }
  }
}

provider "yandex" {
  cloud_id  = var.cloud_id
  folder_id = var.folder_id
  zone      = var.zone
  # Empty: the provider takes YC_TOKEN / YC_SERVICE_ACCOUNT_KEY_FILE from the environment.
  service_account_key_file = var.service_account_key_file != "" ? var.service_account_key_file : null
}
