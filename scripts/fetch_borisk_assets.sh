#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
asset_dir="${1:-results/borisk_assets}"
revision=88f87bc4640a4d8127ad3e170e094e13a78b7fca
base="https://raw.githubusercontent.com/saitcakmak/BoRisk/$revision/BoRisk/test_functions"

fetch() {
    local source="$1" destination="$2"
    if [[ -s "$destination" ]]; then
        return
    fi
    mkdir -p "$(dirname "$destination")"
    curl --fail --silent --show-error --location --retry 2 --max-time 30 "$base/$source" -o "$destination.part.$$"
    mv "$destination.part.$$" "$destination"
}

fetch portfolio_surrogate_state_dict.pt "$asset_dir/portfolio_surrogate_state_dict.pt"
for i in {1..30}; do
    part="port_evals/port_n=100_seed=$i"
    fetch "$part" "$asset_dir/$part"
done

upstream="$asset_dir/upstream/BoRisk/test_functions"
fetch covid_exp_class.py "$upstream/covid_exp_class.py"
for module in __init__ analysis_helpers base_params modified_params run_sims sir_stochastic_sim stochastic_simulation subdivide_severity; do
    path="covid_simulators/$module.py"
    fetch "$path" "$upstream/$path"
done
