"""prior_similarity — measure how far the synthetic training prior
(src/data_gen.py) is from the real ERA5 episode distribution
(eval/data/era5_global_corpus.py), on indicators computed by the SAME code
for both sources.

Both sources are reduced to a `FieldBundle` (bundles.py) before any indicator
(indicators.py) touches them, so no indicator can accidentally see a
source-specific shortcut. The runner is eval/runners/prior_similarity_eval.py.
"""
