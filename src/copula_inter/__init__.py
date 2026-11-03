"""Inter-instance Gaussian copula: data generation, PIT, model and training.

Subpackages: data (synthetic GP episodes and datasets), era5 (real ARCO-ERA5
data), marginal (PIT and Phase-A marginal fine-tuning), copula (model and
losses), baselines (classical GP references), training (the copula training
loop), viz (field figures). The top-level train, generate_pit_dataset,
finetune_marginal and diag_kernels modules are the command-line entry points.
See docs/architecture.md.
"""
