"""BeyondArena extension for joint (multi-row) predictive evaluation.

- joint_metrics: joint scores of a Gaussian-copula predictive over a whole test fold.
- joint_model:  TabArena / AutoGluon model classes (a marginal backend with or without its copula,
                and a random-intercept mixed model) that expose ``predict_joint``.
- joint_runner: the TabArena experiment runner that scores that joint predictive on the full test fold.

joint_model and joint_runner import tabarena / autogluon (the beyondarena-joint venv);
joint_metrics depends only on numpy / torch.
"""
