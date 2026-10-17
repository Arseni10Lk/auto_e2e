# Trial Guide — Running AutoE2E Tests on AWS EC2

This guide walks you through launching a GPU instance, setting up the environment, and running the test suite end-to-end.

## Step 1: Launch an EC2 Instance

### AMI Selection

Use the **Deep Learning OSS Nvidia Driver AMI (Ubuntu 24.04)** provided by AWS.

To find it in the AWS Console:
1. Go to EC2 → Launch Instance → Browse AMIs
2. Search for: `Deep Learning OSS Nvidia Driver AMI GPU PyTorch`
3. Select the latest Ubuntu 24.04 variant (e.g., `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.10 (Ubuntu 24.04)`)

Or via AWS CLI:
```bash
aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.* (Ubuntu 24.04)*" \
            "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].{Name:Name, ImageId:ImageId}' \
  --output table
```

### Instance Type

| Instance Type | GPU | vCPUs | Memory | Cost (us-east-1) |
|---------------|-----|-------|--------|-------------------|
| `g4dn.xlarge` (Recommended) | 1× NVIDIA T4 (16 GB) | 4 | 16 GB | ~$0.526/h |

### Launch (CLI Example)

```bash
aws ec2 run-instances \
  --image-id <ami-id> \
  --instance-type g4dn.xlarge \
  --subnet-id <private-subnet-id> \
  --iam-instance-profile Name=<ssm-instance-profile-name> \
  --no-associate-public-ip-address \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=auto-e2e-test}]'
```

## Step 2: Connect via Session Manager

Once the instance is running and SSM agent is online (~30–60 seconds after launch):

```bash
aws ssm start-session --target <instance-id>
```

Then switch to the ubuntu user:
```bash
sudo -i -u ubuntu
cd ~
```

## Step 3: Clone the Repository

```bash
git clone https://github.com/autowarefoundation/auto_e2e.git
cd auto_e2e
```

## Step 4: Set Up Python Environment

The AMI comes with NVIDIA drivers pre-installed but Python packages are not available globally. Create a virtual environment:

```bash
python3 -m venv ~/e2e-env
source ~/e2e-env/bin/activate
```

Install dependencies with CUDA torch wheels (plain `make setup` installs CPU-only wheels):
```bash
make setup TORCH_CHANNEL=cu118
```

This takes 2–3 minutes. Verify the installation:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch 2.7.1+cu118, CUDA available: True
```

## Step 5: Run the Tests

```bash
cd ~/auto_e2e
make test
```

The run ends with a pytest summary reporting that all selected tests passed.
For the model's inputs, outputs and forward signature, see the
[Model architecture guide](./Model/).

## Step 6: Clean Up

Remember to terminate the instance when you are done to avoid unnecessary charges:

```bash
aws ec2 terminate-instances --instance-ids <instance-id>
```
