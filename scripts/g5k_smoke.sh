#!/bin/bash
#OAR -n g5k_smoke
#OAR -l gpu=1,walltime=0:20:00
# Smoke test of a Grid5000 site for this project: GPU, shared venv, group-storage
# read speed, HF cache, wandb reachability. Submit it where you plan to train:
#     python3 scripts/g5k_gpus.py submit --walltime 0:20 -- scripts/g5k_smoke.sh
set -uo pipefail

t0=$(date +%s)
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
echo "host=$(hostname) site=$(hostname -d | cut -d. -f1) job=${OAR_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

python -c "
import time; t = time.time()
import torch
print(f'torch {torch.__version__} import {time.time() - t:.0f}s, cuda={torch.cuda.is_available()}, '
      f'bf16={torch.cuda.is_bf16_supported()}, device={torch.cuda.get_device_name(0)}')
x = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
y = x @ x  # warm-up: first cuBLAS call loads kernels
torch.cuda.synchronize(); t = time.time()
for _ in range(20): y = x @ x
torch.cuda.synchronize(); dt = time.time() - t
print(f'fp16 matmul {20 * 2 * 8192**3 / dt / 1e12:.0f} TFLOP/s')
import copula_inter; print('copula_inter import ok')
"
echo "env setup + imports: $(( $(date +%s) - t0 ))s"

big=$(find checkpoints -name '*.pt' -size +300M 2>/dev/null | head -1)
if [[ -n "$big" ]]; then
    echo "group-storage read ($big):"
    dd if="$big" of=/dev/null bs=16M count=64 iflag=direct 2>&1 | tail -1
fi
echo "HF_HOME=${HF_HOME:-unset} ($(du -sh "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null | cut -f1))"
echo "wandb api: HTTP $(curl -s -o /dev/null -m 15 -w '%{http_code}' https://api.wandb.ai)"
echo "huggingface: HTTP $(curl -s -o /dev/null -m 15 -w '%{http_code}' https://huggingface.co)"
[[ -f ~/.netrc ]] && echo "~/.netrc (wandb) present" || echo "~/.netrc (wandb) MISSING: g5k_gpus.py setup-site <site> --copy-secrets"
for f in .env ~/.config/copula-inter/tabpfn_token ~/.config/tabpfn_token; do [[ -f $f ]] && echo "TabPFN key source present: $f"; done
echo "total ${SECONDS}s"
