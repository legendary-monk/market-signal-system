# market-signal-system
Daily Nifty signal bot

## Advanced Internal Analysis (math + physics)

The system now includes:
- Kalman state-space latent drift/acceleration estimation (`models/state_space.py`)
- Regime probability inference for diffusion/trend/shock (`models/regime_hmm.py`)
- Physics diagnostics: market energy + entropy (`models/market_physics.py`)
- Bayesian probability fusion (`models/fusion.py`)
- Constrained fractional Kelly sizing (`risk/sizing.py`)
