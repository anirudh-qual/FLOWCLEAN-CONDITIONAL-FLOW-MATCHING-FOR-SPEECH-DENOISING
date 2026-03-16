# FlowClean: Conditional Flow Matching for Speech Denoising

## Setup

```bash
# Create conda env
conda create -n flowclean python=3.11 -y
conda activate flowclean

# Install PyTorch (CUDA 12.1)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install "datasets<4.0" librosa soundfile pyyaml pesq pystoi

# (Optional) Install wandb for experiment tracking
pip install wandb
```

## HuggingFace Token

The dataset is loaded from HuggingFace (`JacobLinCool/VoiceBank-DEMAND-16k`). If you hit rate limits or access issues:

```bash
# Login once
huggingface-cli login
# OR set the env var
export HF_TOKEN=your_token_here
```

## Train

```bash
conda activate flowclean
python train.py --config configs/default.yaml
```

Checkpoints are saved to `./checkpoints/`. Edit `configs/default.yaml` to change hyperparameters.

### With Wandb

In `configs/default.yaml`, set:

```yaml
wandb:
  use_wandb: true
  wandb_token: "your_api_key"   # or set WANDB_API_KEY env var
```

Then run training as usual. Logs `train/loss`, `train/loss_fm`, `train/loss_mr`, `train/lr` per step, and `epoch/avg_loss` per epoch. Inference logs `eval/pesq` and `eval/stoi`.

## Inference

```bash
python inference.py --checkpoint checkpoints/flowclean_best.pt \
                    --ode_steps 10 \
                    --solver euler \
                    --output_dir ./enhanced
```

Add `--eval_metrics` to compute PESQ and STOI on the test set.

## Project Structure

```
flowclean/
  config.py          # FlowCleanConfig dataclass
  models/unet.py     # Conditional U-Net backbone
  data/              # VoiceBank-DEMAND HF dataset loader
  losses/stft_loss.py# Multi-resolution STFT loss
  utils/stft.py      # STFT / iSTFT utilities
configs/default.yaml # All hyperparameters
train.py             # Training script
inference.py         # Inference + evaluation
```
