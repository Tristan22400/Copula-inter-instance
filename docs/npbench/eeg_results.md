# EEG joint log-likelihood: TFM marginals, independent vs copula vs autoregressive

Nats per target scalar, higher is better. Our rows are zero-shot (pretrained marginals, copulas trained on
synthetic GP data); the published models were trained on the EEG training subjects. indep and copula rows
cover every task; AR rows cover an evenly spread subset (n), and [same eps: ...] gives indep and copula
on exactly those episodes.

## Bruinsma et al. (2023) EEG, exact protocol (mean ± 1.96 SE, raw µV)

| Model | Int. | Rec. |
|---|---|---|
| ConvCNP (published) | -1.02 ± 0.01 | -2.07 ± 0.03 |
| ConvCNP (AR) (published) | -0.34 ± 0.01 | -0.63 ± 0.01 |
| ConvGNP (published) | -0.93 ± 0.01 | -1.45 ± 0.03 |
| FullConvGNP (published) | -0.35 ± 0.01 | -0.57 ± 0.01 |
| ConvLNP (ML) (published) | -1.04 ± 0.01 | -1.53 ± 0.02 |
| ConvLNP (ELBO) (published) | -1.20 ± 0.01 | -2.00 ± 0.06 |
| TabICLv2 + indep | -1.66 ± 0.01 | -2.58 ± 0.03 |
| TabICLv2 + copula | -1.62 ± 0.01 | -2.34 ± 0.02 |
| TabICLv2 + AR (n=64/64) | -1.00 ± 0.06 [same eps: indep -1.65, copula -1.61] | -1.12 ± 0.12 [same eps: indep -2.33, copula -2.19] |
| TabPFN-3.5 + indep | -1.78 ± 0.01 | -2.38 ± 0.02 |
| TabPFN-3.5 + copula | -1.74 ± 0.01 | -2.16 ± 0.02 |
| TabPFN-3.5 + AR (n=64/64) | -1.20 ± 0.11 [same eps: indep -1.71, copula -1.68] | -1.19 ± 0.18 [same eps: indep -2.10, copula -1.98] |
| Kumo + indep | -1.78 ± 0.01 | -2.59 ± 0.03 |
| Kumo + copula | -1.74 ± 0.01 | -2.29 ± 0.02 |
| Kumo + AR (n=64/64) | -1.14 ± 0.05 [same eps: indep -1.77, copula -1.74] | -1.19 ± 0.13 [same eps: indep -2.31, copula -2.10] |

## Hassan et al. (2026) EEG, protocol reconstructed (mean (SEM) over trials × N, standardised channels)

| Model | EEG-Int (M=16) | EEG-For (M=16) | EEG-Int (M=64) | EEG-For (M=64) |
|---|---|---|---|---|
| TNP-D (AR) (published) | 0.51 (0.013) | 1.07 (0.004) | 0.88 (0.011) | 1.12 (0.003) |
| TNP-D (Ind) (published) | 0.36 (0.014) | -0.74 (0.008) | 0.35 (0.014) | -1.08 (0.007) |
| TNP-ND (published) | 0.46 (0.011) | -0.04 (0.005) | 0.50 (0.010) | -0.23 (0.004) |
| TNP-A (published) | 0.58 (0.014) | 1.23 (0.003) | 0.95 (0.012) | 1.20 (0.003) |
| TNP buffer K=16 (published) | 0.52 (0.013) | 0.85 (0.004) | 0.90 (0.011) | 1.12 (0.003) |
| TNP buffer K=4 (published) | 0.54 (0.014) | 1.17 (0.003) | 0.91 (0.011) | 1.18 (0.003) |
| TNP buffer K=1 (published) | 0.54 (0.014) | 1.21 (0.003) | 0.91 (0.011) | 1.19 (0.003) |
| TabICLv2 + indep | -0.11 (0.012) | -1.14 (0.011) | -0.11 (0.012) | -1.51 (0.012) |
| TabICLv2 + copula | -0.07 (0.012) | -0.72 (0.006) | -0.05 (0.011) | -0.95 (0.005) |
| TabICLv2 + AR (n=192/192/192/192) | 0.41 (0.049) [same eps: indep -0.09, copula -0.06] | 0.63 (0.040) [same eps: indep -1.11, copula -0.71] | 0.60 (0.046) [same eps: indep -0.10, copula -0.05] | 0.17 (0.080) [same eps: indep -1.60, copula -0.92] |
| TabPFN-3.5 + indep | -0.31 (0.010) | -1.17 (0.013) | -0.31 (0.010) | -1.45 (0.012) |
| TabPFN-3.5 + copula | -0.27 (0.010) | -0.55 (0.007) | -0.25 (0.010) | -0.86 (0.006) |
| TabPFN-3.5 + AR (n=192/192/192/192) | 0.26 (0.041) [same eps: indep -0.27, copula -0.23] | 0.62 (0.028) [same eps: indep -1.14, copula -0.54] | 0.47 (0.037) [same eps: indep -0.27, copula -0.21] | 0.66 (0.024) [same eps: indep -1.47, copula -0.83] |
| Kumo + indep | -0.28 (0.011) | -1.16 (0.010) | -0.28 (0.011) | -1.52 (0.009) |
| Kumo + copula | -0.25 (0.011) | -0.74 (0.006) | -0.23 (0.011) | -1.02 (0.005) |
| Kumo + AR (n=192/192/192/192) | 0.37 (0.044) [same eps: indep -0.27, copula -0.24] | 0.78 (0.018) [same eps: indep -1.17, copula -0.74] | 0.55 (0.039) [same eps: indep -0.28, copula -0.23] | 0.77 (0.016) [same eps: indep -1.55, copula -1.01] |

## Copula vs independence, paired per episode

| Unit | Marginal | n | indep | copula | copula − indep (± 1.96 SE) | win rate | z_test std |
|---|---|---|---|---|---|---|---|
| arcnp_interpolation | Kumo | 4096 | -1.779 | -1.744 | +0.034 ± 0.002 | 0.93 | 1.04 |
| arcnp_interpolation | TabICLv2 | 4096 | -1.657 | -1.619 | +0.038 ± 0.002 | 0.91 | 1.06 |
| arcnp_interpolation | TabPFN-3.5 | 4096 | -1.783 | -1.742 | +0.041 ± 0.001 | 0.98 | 0.96 |
| arcnp_reconstruction | Kumo | 4096 | -2.585 | -2.292 | +0.293 ± 0.013 | 0.96 | 0.88 |
| arcnp_reconstruction | TabICLv2 | 4096 | -2.578 | -2.337 | +0.241 ± 0.012 | 0.94 | 0.95 |
| arcnp_reconstruction | TabPFN-3.5 | 4096 | -2.378 | -2.155 | +0.223 ± 0.009 | 0.96 | 0.75 |
| hassan_for_N8_M16 | Kumo | 897 | -1.494 | -0.749 | +0.745 ± 0.047 | 0.99 | 1.11 |
| hassan_for_N8_M16 | TabICLv2 | 897 | -1.492 | -0.690 | +0.802 ± 0.059 | 1.00 | 0.92 |
| hassan_for_N8_M16 | TabPFN-3.5 | 897 | -1.640 | -0.663 | +0.977 ± 0.061 | 0.98 | 1.22 |
| hassan_for_N16_M16 | Kumo | 897 | -1.238 | -0.692 | +0.546 ± 0.035 | 0.99 | 0.95 |
| hassan_for_N16_M16 | TabICLv2 | 897 | -1.404 | -0.796 | +0.608 ± 0.045 | 1.00 | 0.60 |
| hassan_for_N16_M16 | TabPFN-3.5 | 897 | -1.256 | -0.526 | +0.730 ± 0.049 | 0.96 | 1.09 |
| hassan_for_N32_M16 | Kumo | 897 | -1.150 | -0.683 | +0.467 ± 0.031 | 0.99 | 0.85 |
| hassan_for_N32_M16 | TabICLv2 | 897 | -1.194 | -0.716 | +0.478 ± 0.035 | 1.00 | 0.61 |
| hassan_for_N32_M16 | TabPFN-3.5 | 897 | -1.190 | -0.514 | +0.676 ± 0.046 | 0.95 | 1.05 |
| hassan_for_N64_M16 | Kumo | 897 | -1.049 | -0.719 | +0.330 ± 0.022 | 0.99 | 0.75 |
| hassan_for_N64_M16 | TabICLv2 | 897 | -0.983 | -0.693 | +0.290 ± 0.022 | 1.00 | 0.62 |
| hassan_for_N64_M16 | TabPFN-3.5 | 897 | -1.069 | -0.539 | +0.530 ± 0.040 | 0.95 | 1.03 |
| hassan_for_N128_M16 | Kumo | 897 | -0.978 | -0.747 | +0.231 ± 0.016 | 0.99 | 0.72 |
| hassan_for_N128_M16 | TabICLv2 | 897 | -0.865 | -0.693 | +0.172 ± 0.014 | 1.00 | 0.60 |
| hassan_for_N128_M16 | TabPFN-3.5 | 897 | -0.924 | -0.509 | +0.415 ± 0.032 | 0.94 | 0.95 |
| hassan_for_N192_M16 | Kumo | 897 | -1.035 | -0.826 | +0.210 ± 0.018 | 0.98 | 0.73 |
| hassan_for_N192_M16 | TabICLv2 | 897 | -0.885 | -0.753 | +0.132 ± 0.016 | 0.99 | 0.62 |
| hassan_for_N192_M16 | TabPFN-3.5 | 897 | -0.946 | -0.572 | +0.374 ± 0.032 | 0.94 | 0.91 |
| hassan_for_N8_M64 | Kumo | 897 | -1.943 | -1.090 | +0.853 ± 0.053 | 0.99 | 1.29 |
| hassan_for_N8_M64 | TabICLv2 | 897 | -1.906 | -0.844 | +1.062 ± 0.082 | 0.99 | 1.25 |
| hassan_for_N8_M64 | TabPFN-3.5 | 897 | -1.990 | -1.098 | +0.893 ± 0.061 | 0.96 | 1.37 |
| hassan_for_N16_M64 | Kumo | 897 | -1.576 | -1.019 | +0.557 ± 0.035 | 0.99 | 1.08 |
| hassan_for_N16_M64 | TabICLv2 | 897 | -1.587 | -0.814 | +0.773 ± 0.060 | 1.00 | 1.01 |
| hassan_for_N16_M64 | TabPFN-3.5 | 897 | -1.537 | -0.893 | +0.644 ± 0.047 | 0.95 | 1.21 |
| hassan_for_N32_M64 | Kumo | 897 | -1.444 | -0.996 | +0.448 ± 0.028 | 0.99 | 0.97 |
| hassan_for_N32_M64 | TabICLv2 | 897 | -1.428 | -0.912 | +0.516 ± 0.035 | 1.00 | 0.79 |
| hassan_for_N32_M64 | TabPFN-3.5 | 897 | -1.377 | -0.814 | +0.563 ± 0.038 | 0.97 | 1.15 |
| hassan_for_N64_M64 | Kumo | 897 | -1.402 | -0.982 | +0.420 ± 0.027 | 1.00 | 0.85 |
| hassan_for_N64_M64 | TabICLv2 | 897 | -1.449 | -1.028 | +0.421 ± 0.030 | 1.00 | 0.62 |
| hassan_for_N64_M64 | TabPFN-3.5 | 897 | -1.305 | -0.767 | +0.539 ± 0.038 | 0.98 | 1.06 |
| hassan_for_N128_M64 | Kumo | 897 | -1.348 | -0.986 | +0.362 ± 0.023 | 1.00 | 0.84 |
| hassan_for_N128_M64 | TabICLv2 | 897 | -1.355 | -1.028 | +0.326 ± 0.024 | 1.00 | 0.62 |
| hassan_for_N128_M64 | TabPFN-3.5 | 897 | -1.209 | -0.746 | +0.463 ± 0.033 | 0.97 | 0.99 |
| hassan_for_N192_M64 | Kumo | 897 | -1.424 | -1.066 | +0.358 ± 0.028 | 1.00 | 0.85 |
| hassan_for_N192_M64 | TabICLv2 | 897 | -1.365 | -1.076 | +0.289 ± 0.031 | 1.00 | 0.65 |
| hassan_for_N192_M64 | TabPFN-3.5 | 897 | -1.269 | -0.821 | +0.448 ± 0.037 | 0.98 | 0.97 |
| hassan_int_N8_M16 | Kumo | 897 | -1.209 | -1.100 | +0.109 ± 0.011 | 0.82 | 1.22 |
| hassan_int_N8_M16 | TabICLv2 | 897 | -1.105 | -0.972 | +0.134 ± 0.014 | 0.78 | 1.08 |
| hassan_int_N8_M16 | TabPFN-3.5 | 897 | -1.089 | -0.976 | +0.113 ± 0.011 | 0.82 | 1.02 |
| hassan_int_N16_M16 | Kumo | 897 | -0.916 | -0.873 | +0.043 ± 0.005 | 0.79 | 1.00 |
| hassan_int_N16_M16 | TabICLv2 | 897 | -0.788 | -0.737 | +0.052 ± 0.008 | 0.70 | 1.03 |
| hassan_int_N16_M16 | TabPFN-3.5 | 897 | -0.920 | -0.854 | +0.066 ± 0.006 | 0.82 | 0.94 |
| hassan_int_N32_M16 | Kumo | 897 | -0.647 | -0.628 | +0.019 ± 0.003 | 0.65 | 0.97 |
| hassan_int_N32_M16 | TabICLv2 | 897 | -0.490 | -0.477 | +0.014 ± 0.004 | 0.58 | 0.97 |
| hassan_int_N32_M16 | TabPFN-3.5 | 897 | -0.668 | -0.636 | +0.032 ± 0.004 | 0.74 | 0.90 |
| hassan_int_N64_M16 | Kumo | 897 | -0.242 | -0.232 | +0.009 ± 0.002 | 0.56 | 0.99 |
| hassan_int_N64_M16 | TabICLv2 | 897 | -0.071 | -0.066 | +0.005 ± 0.002 | 0.56 | 0.91 |
| hassan_int_N64_M16 | TabPFN-3.5 | 897 | -0.202 | -0.190 | +0.013 ± 0.002 | 0.63 | 0.87 |
| hassan_int_N128_M16 | Kumo | 897 | 0.397 | 0.402 | +0.005 ± 0.001 | 0.61 | 0.89 |
| hassan_int_N128_M16 | TabICLv2 | 897 | 0.635 | 0.637 | +0.002 ± 0.001 | 0.57 | 0.86 |
| hassan_int_N128_M16 | TabPFN-3.5 | 897 | 0.347 | 0.359 | +0.012 ± 0.001 | 0.76 | 0.76 |
| hassan_int_N192_M16 | Kumo | 897 | 0.928 | 0.932 | +0.003 ± 0.001 | 0.58 | 0.85 |
| hassan_int_N192_M16 | TabICLv2 | 897 | 1.178 | 1.179 | +0.001 ± 0.001 | 0.52 | 0.84 |
| hassan_int_N192_M16 | TabPFN-3.5 | 897 | 0.665 | 0.680 | +0.015 ± 0.001 | 0.89 | 0.67 |
| hassan_int_N8_M64 | Kumo | 897 | -1.204 | -1.042 | +0.163 ± 0.011 | 0.94 | 1.24 |
| hassan_int_N8_M64 | TabICLv2 | 897 | -1.102 | -0.909 | +0.194 ± 0.013 | 0.91 | 1.11 |
| hassan_int_N8_M64 | TabPFN-3.5 | 897 | -1.083 | -0.933 | +0.149 ± 0.009 | 0.92 | 1.05 |
| hassan_int_N16_M64 | Kumo | 897 | -0.920 | -0.852 | +0.067 ± 0.005 | 0.91 | 1.03 |
| hassan_int_N16_M64 | TabICLv2 | 897 | -0.806 | -0.715 | +0.091 ± 0.008 | 0.83 | 1.07 |
| hassan_int_N16_M64 | TabPFN-3.5 | 897 | -0.913 | -0.819 | +0.094 ± 0.005 | 0.94 | 0.97 |
| hassan_int_N32_M64 | Kumo | 897 | -0.652 | -0.622 | +0.031 ± 0.002 | 0.84 | 0.99 |
| hassan_int_N32_M64 | TabICLv2 | 897 | -0.497 | -0.458 | +0.039 ± 0.004 | 0.79 | 0.99 |
| hassan_int_N32_M64 | TabPFN-3.5 | 897 | -0.670 | -0.619 | +0.050 ± 0.003 | 0.91 | 0.92 |
| hassan_int_N64_M64 | Kumo | 897 | -0.242 | -0.229 | +0.013 ± 0.002 | 0.72 | 1.00 |
| hassan_int_N64_M64 | TabICLv2 | 897 | -0.069 | -0.054 | +0.015 ± 0.002 | 0.76 | 0.92 |
| hassan_int_N64_M64 | TabPFN-3.5 | 897 | -0.202 | -0.180 | +0.022 ± 0.002 | 0.85 | 0.89 |
| hassan_int_N128_M64 | Kumo | 897 | 0.399 | 0.407 | +0.008 ± 0.001 | 0.78 | 0.90 |
| hassan_int_N128_M64 | TabICLv2 | 897 | 0.631 | 0.638 | +0.006 ± 0.001 | 0.75 | 0.87 |
| hassan_int_N128_M64 | TabPFN-3.5 | 897 | 0.354 | 0.373 | +0.018 ± 0.001 | 0.95 | 0.77 |
| hassan_int_N192_M64 | Kumo | 897 | 0.935 | 0.939 | +0.004 ± 0.000 | 0.78 | 0.86 |
| hassan_int_N192_M64 | TabICLv2 | 897 | 1.182 | 1.186 | +0.004 ± 0.000 | 0.75 | 0.84 |
| hassan_int_N192_M64 | TabPFN-3.5 | 897 | 0.673 | 0.698 | +0.025 ± 0.001 | 0.99 | 0.68 |

## Copula vs autoregressive, same marginal, on the AR episodes

| Unit | Marginal | n (AR episodes) | indep | copula | AR | copula − AR (± 1.96 SE) |
|---|---|---|---|---|---|---|
| arcnp_interpolation | Kumo | 64 | -1.766 | -1.736 | -1.142 | -0.594 ± 0.040 |
| arcnp_interpolation | TabICLv2 | 64 | -1.645 | -1.610 | -1.004 | -0.606 ± 0.033 |
| arcnp_interpolation | TabPFN-3.5 | 64 | -1.709 | -1.675 | -1.202 | -0.473 ± 0.027 |
| arcnp_reconstruction | Kumo | 64 | -2.308 | -2.101 | -1.185 | -0.916 ± 0.078 |
| arcnp_reconstruction | TabICLv2 | 64 | -2.331 | -2.185 | -1.121 | -1.064 ± 0.083 |
| arcnp_reconstruction | TabPFN-3.5 | 64 | -2.104 | -1.975 | -1.186 | -0.789 ± 0.074 |
| hassan_for_N8_M16 | Kumo | 32 | -1.393 | -0.691 | 0.727 | -1.418 ± 0.124 |
| hassan_for_N8_M16 | TabICLv2 | 32 | -1.198 | -0.623 | 0.245 | -0.867 ± 0.250 |
| hassan_for_N8_M16 | TabPFN-3.5 | 32 | -1.356 | -0.551 | 0.560 | -1.111 ± 0.118 |
| hassan_for_N16_M16 | Kumo | 32 | -1.197 | -0.652 | 0.766 | -1.418 ± 0.106 |
| hassan_for_N16_M16 | TabICLv2 | 32 | -1.157 | -0.738 | 0.430 | -1.168 ± 0.175 |
| hassan_for_N16_M16 | TabPFN-3.5 | 32 | -1.189 | -0.416 | 0.592 | -1.009 ± 0.083 |
| hassan_for_N32_M16 | Kumo | 32 | -1.173 | -0.731 | 0.807 | -1.538 ± 0.089 |
| hassan_for_N32_M16 | TabICLv2 | 32 | -1.440 | -0.769 | 0.616 | -1.385 ± 0.153 |
| hassan_for_N32_M16 | TabPFN-3.5 | 32 | -1.219 | -0.608 | 0.605 | -1.213 ± 0.123 |
| hassan_for_N64_M16 | Kumo | 32 | -1.223 | -0.785 | 0.873 | -1.658 ± 0.084 |
| hassan_for_N64_M16 | TabICLv2 | 32 | -1.024 | -0.707 | 0.807 | -1.514 ± 0.102 |
| hassan_for_N64_M16 | TabPFN-3.5 | 32 | -1.197 | -0.579 | 0.685 | -1.264 ± 0.117 |
| hassan_for_N128_M16 | Kumo | 32 | -0.912 | -0.673 | 0.775 | -1.448 ± 0.093 |
| hassan_for_N128_M16 | TabICLv2 | 32 | -0.782 | -0.610 | 0.871 | -1.481 ± 0.096 |
| hassan_for_N128_M16 | TabPFN-3.5 | 32 | -0.834 | -0.444 | 0.651 | -1.095 ± 0.117 |
| hassan_for_N192_M16 | Kumo | 32 | -1.116 | -0.883 | 0.754 | -1.637 ± 0.153 |
| hassan_for_N192_M16 | TabICLv2 | 32 | -1.047 | -0.843 | 0.834 | -1.678 ± 0.142 |
| hassan_for_N192_M16 | TabPFN-3.5 | 32 | -1.026 | -0.636 | 0.632 | -1.267 ± 0.143 |
| hassan_for_N8_M64 | Kumo | 32 | -2.031 | -1.022 | 0.748 | -1.770 ± 0.096 |
| hassan_for_N8_M64 | TabICLv2 | 32 | -2.008 | -0.769 | -0.795 | +0.027 ± 0.526 |
| hassan_for_N8_M64 | TabPFN-3.5 | 32 | -2.084 | -1.014 | 0.593 | -1.607 ± 0.132 |
| hassan_for_N16_M64 | Kumo | 32 | -1.608 | -1.019 | 0.774 | -1.793 ± 0.102 |
| hassan_for_N16_M64 | TabICLv2 | 32 | -1.732 | -0.808 | -0.348 | -0.460 ± 0.383 |
| hassan_for_N16_M64 | TabPFN-3.5 | 32 | -1.607 | -0.841 | 0.623 | -1.464 ± 0.113 |
| hassan_for_N32_M64 | Kumo | 32 | -1.467 | -1.003 | 0.810 | -1.813 ± 0.085 |
| hassan_for_N32_M64 | TabICLv2 | 32 | -1.580 | -0.909 | 0.180 | -1.090 ± 0.329 |
| hassan_for_N32_M64 | TabPFN-3.5 | 32 | -1.466 | -0.840 | 0.659 | -1.499 ± 0.094 |
| hassan_for_N64_M64 | Kumo | 32 | -1.464 | -1.021 | 0.828 | -1.849 ± 0.099 |
| hassan_for_N64_M64 | TabICLv2 | 32 | -1.410 | -0.997 | 0.661 | -1.659 ± 0.162 |
| hassan_for_N64_M64 | TabPFN-3.5 | 32 | -1.275 | -0.778 | 0.700 | -1.478 ± 0.114 |
| hassan_for_N128_M64 | Kumo | 32 | -1.355 | -0.952 | 0.774 | -1.725 ± 0.124 |
| hassan_for_N128_M64 | TabICLv2 | 32 | -1.457 | -0.989 | 0.650 | -1.639 ± 0.141 |
| hassan_for_N128_M64 | TabPFN-3.5 | 32 | -1.242 | -0.695 | 0.704 | -1.399 ± 0.131 |
| hassan_for_N192_M64 | Kumo | 32 | -1.361 | -1.038 | 0.703 | -1.740 ± 0.150 |
| hassan_for_N192_M64 | TabICLv2 | 32 | -1.402 | -1.063 | 0.661 | -1.724 ± 0.184 |
| hassan_for_N192_M64 | TabPFN-3.5 | 32 | -1.174 | -0.794 | 0.667 | -1.461 ± 0.136 |
| hassan_int_N8_M16 | Kumo | 32 | -1.271 | -1.145 | -0.248 | -0.897 ± 0.095 |
| hassan_int_N8_M16 | TabICLv2 | 32 | -1.127 | -0.986 | -0.345 | -0.642 ± 0.123 |
| hassan_int_N8_M16 | TabPFN-3.5 | 32 | -1.097 | -0.981 | -0.304 | -0.677 ± 0.074 |
| hassan_int_N16_M16 | Kumo | 32 | -0.911 | -0.883 | -0.118 | -0.766 ± 0.083 |
| hassan_int_N16_M16 | TabICLv2 | 32 | -0.761 | -0.716 | -0.112 | -0.604 ± 0.089 |
| hassan_int_N16_M16 | TabPFN-3.5 | 32 | -0.893 | -0.827 | -0.171 | -0.656 ± 0.079 |
| hassan_int_N32_M16 | Kumo | 32 | -0.595 | -0.578 | 0.103 | -0.681 ± 0.074 |
| hassan_int_N32_M16 | TabICLv2 | 32 | -0.440 | -0.412 | 0.121 | -0.533 ± 0.078 |
| hassan_int_N32_M16 | TabPFN-3.5 | 32 | -0.625 | -0.587 | 0.054 | -0.640 ± 0.079 |
| hassan_int_N64_M16 | Kumo | 32 | -0.197 | -0.194 | 0.372 | -0.567 ± 0.079 |
| hassan_int_N64_M16 | TabICLv2 | 32 | -0.034 | -0.029 | 0.419 | -0.448 ± 0.066 |
| hassan_int_N64_M16 | TabPFN-3.5 | 32 | -0.125 | -0.119 | 0.316 | -0.435 ± 0.062 |
| hassan_int_N128_M16 | Kumo | 32 | 0.391 | 0.396 | 0.830 | -0.433 ± 0.072 |
| hassan_int_N128_M16 | TabICLv2 | 32 | 0.626 | 0.628 | 0.967 | -0.339 ± 0.066 |
| hassan_int_N128_M16 | TabPFN-3.5 | 32 | 0.409 | 0.421 | 0.703 | -0.282 ± 0.054 |
| hassan_int_N192_M16 | Kumo | 32 | 0.974 | 0.976 | 1.253 | -0.277 ± 0.048 |
| hassan_int_N192_M16 | TabICLv2 | 32 | 1.179 | 1.180 | 1.395 | -0.214 ± 0.035 |
| hassan_int_N192_M16 | TabPFN-3.5 | 32 | 0.703 | 0.718 | 0.977 | -0.259 ± 0.046 |
| hassan_int_N8_M64 | Kumo | 32 | -1.224 | -1.044 | 0.050 | -1.095 ± 0.121 |
| hassan_int_N8_M64 | TabICLv2 | 32 | -1.105 | -0.929 | -0.105 | -0.824 ± 0.134 |
| hassan_int_N8_M64 | TabPFN-3.5 | 32 | -1.075 | -0.937 | 0.033 | -0.970 ± 0.103 |
| hassan_int_N16_M64 | Kumo | 32 | -0.912 | -0.842 | 0.128 | -0.970 ± 0.075 |
| hassan_int_N16_M64 | TabICLv2 | 32 | -0.788 | -0.678 | 0.150 | -0.828 ± 0.078 |
| hassan_int_N16_M64 | TabPFN-3.5 | 32 | -0.880 | -0.797 | 0.116 | -0.913 ± 0.064 |
| hassan_int_N32_M64 | Kumo | 32 | -0.656 | -0.631 | 0.277 | -0.908 ± 0.091 |
| hassan_int_N32_M64 | TabICLv2 | 32 | -0.491 | -0.472 | 0.302 | -0.774 ± 0.086 |
| hassan_int_N32_M64 | TabPFN-3.5 | 32 | -0.646 | -0.593 | 0.250 | -0.844 ± 0.066 |
| hassan_int_N64_M64 | Kumo | 32 | -0.225 | -0.205 | 0.515 | -0.720 ± 0.071 |
| hassan_int_N64_M64 | TabICLv2 | 32 | -0.067 | -0.046 | 0.614 | -0.660 ± 0.060 |
| hassan_int_N64_M64 | TabPFN-3.5 | 32 | -0.138 | -0.120 | 0.478 | -0.597 ± 0.058 |
| hassan_int_N128_M64 | Kumo | 32 | 0.410 | 0.418 | 0.980 | -0.562 ± 0.068 |
| hassan_int_N128_M64 | TabICLv2 | 32 | 0.647 | 0.654 | 1.130 | -0.476 ± 0.056 |
| hassan_int_N128_M64 | TabPFN-3.5 | 32 | 0.408 | 0.426 | 0.835 | -0.409 ± 0.052 |
| hassan_int_N192_M64 | Kumo | 32 | 0.943 | 0.948 | 1.333 | -0.384 ± 0.055 |
| hassan_int_N192_M64 | TabICLv2 | 32 | 1.181 | 1.185 | 1.500 | -0.315 ± 0.042 |
| hassan_int_N192_M64 | TabPFN-3.5 | 32 | 0.726 | 0.751 | 1.086 | -0.335 ± 0.054 |

## Marginal preservation (copula draws vs independent draws, KS of Φ(z_i) to U(0,1))

| Marginal | episodes | mean KS, copula draws | mean KS, independent draws | max abs(diag Σ − 1) |
|---|---|---|---|---|
| TabICLv2 | 832 | 0.0380 | 0.0382 | 2.4e-07 |
| TabPFN-3.5 | 832 | 0.0380 | 0.0382 | 2.4e-07 |
| Kumo | 832 | 0.0380 | 0.0382 | 1.8e-07 |

## Joint sampling cost (Hassan EEG-Int, N = 128 context time points)

| Marginal | GPU | M (time pts) | target rows | B × S | indep (s) | copula (s) | AR (s) | AR / copula |
|---|---|---|---|---|---|---|---|---|
| Kumo | NVIDIA A40 | 16 | 112 | 8 × 8 | 0.135 | 0.969 | 223.81 | 231× |
| Kumo | NVIDIA A40 | 64 | 448 | 8 × 8 | 0.173 | 1.061 | 1087.33 | 1025× |
| TabICLv2 | NVIDIA A100-SXM4-40GB | 16 | 112 | 8 × 8 | 0.121 | 0.972 | 111.50 | 115× |
| TabICLv2 | NVIDIA A100-SXM4-40GB | 64 | 448 | 8 × 8 | 0.152 | 1.035 | 599.42 | 579× |
| TabPFN-3.5 | NVIDIA A100-SXM4-40GB | 16 | 112 | 8 × 8 | 0.154 | 1.284 | 245.34 | 191× |
| TabPFN-3.5 | NVIDIA A100-SXM4-40GB | 64 | 448 | 8 × 8 | 0.194 | 1.364 | 1184.26 | 868× |
