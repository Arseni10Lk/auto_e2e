terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Separate state key from the main Platform root (infra/terraform.tfstate).
  # This root only manages console-dedicated resources (ALB SG, CloudFront), so
  # `terraform apply` here cannot plan a change to existing Platform infra.
  # Same bucket and lock table as Platform.
  backend "s3" {
    bucket         = "auto-e2e-platform-tfstate"
    key            = "infra-console/terraform.tfstate"
    region         = "us-east-1"
    profile        = "autowarefoundation"
    dynamodb_table = "auto-e2e-platform-tflock"
    encrypt        = true
  }
}

provider "aws" {
  profile             = "autowarefoundation"
  region              = var.aws_region
  allowed_account_ids = [var.expected_aws_account_id]
}
