# Market Signal System

A free, daily market-signal pipeline for **NIFTY 50** that combines:
- market structure features,
- RSS/news sentiment,
- internal probabilistic models (state-space + regime + fusion),
- physics-inspired stability diagnostics,
- conservative risk sizing,
- and Telegram delivery.

The project is designed to stay low-cost, interpretable, and operationally simple.

---

## 1) What this system does

Each run produces one decision package:
- **Signal**: `BULLISH`, `BEARISH`, or `NEUTRAL`
- **Confidence**: bounded confidence score
- **Posterior probability**: fused `P(up)`
- **Diagnostics**: regime, uncertainty, disagreement, energy, entropy
- **Recommended weight**: constrained fractional Kelly output
- **Narrative**: top reasons and scenario playbook

Outputs are:
1. saved to `predictions.csv`,
2. logged for traceability,
3. sent to Telegram as a formatted report.

---

## 2) Repository structure

```text
market-signal-system/
├── main.py                 # Orchestrates full daily run
├── config.py               # Runtime configuration + validations
├── logger.py               # Logging setup
├── market_data.py          # OHLCV fetch + schema validation
├── news_fetcher.py         # RSS fetch + freshness filtering
├── sentiment.py            # Sentiment scoring
├── features.py             # Market feature engineering
├── signal_engine.py        # Core signal logic + model integration
├── validator.py            # Persist predictions + outcome resolution
├── telegram_bot.py         # Telegram message formatting + sending
├── predictions.csv         # Local signal history
├── models/
│   ├── state_space.py      # Latent drift/acceleration + p_up_1d
│   ├── regime_hmm.py       # Diffusion/trend/shock probabilities
│   ├── fusion.py           # Bayesian/logit fusion
│   └── market_physics.py   # Energy + entropy metrics
└── risk/
    └── sizing.py           # Constrained fractional Kelly sizing
```

---

## 3) End-to-end flow

1. `market_data.fetch_market_data()` downloads NIFTY OHLCV, normalizes column names, validates schema.
2. `news_fetcher.fetch_news()` collects RSS items and drops stale/undated entries (unless explicitly allowed).
3. `sentiment.analyze_sentiment()` scores news polarity.
4. `features.compute_features()` builds trend/momentum/volatility/RSI context.
5. `signal_engine.generate_signal()` computes:
   - base heuristic score,
   - latent state estimate,
   - regime probabilities,
   - physics diagnostics,
   - posterior fused probability,
   - confidence and recommended weight.
6. `validator.save_prediction()` appends a row to `predictions.csv`.
7. `telegram_bot.send_daily_report()` sends the decision report to Telegram.

---

## 4) Core quantitative components

### 4.1 State-space model (`models/state_space.py`)
Estimates latent market state:
- drift,
- acceleration,
- volatility proxy,
- one-day directional probability (`p_up_1d`).

Purpose: smooth noisy price movement into a more stable directional prior.

### 4.2 Regime inference (`models/regime_hmm.py`)
Infers probability of three market regimes:
- diffusion (calm/chop),
- trend (persistent direction),
- shock (dislocation risk).

Purpose: condition confidence/sizing on market regime risk.

### 4.3 Bayesian fusion (`models/fusion.py`)
Combines:
- price probability,
- sentiment probability,
- regime-transformed probability,
into a posterior `P(up)` with disagreement/uncertainty diagnostics.

Purpose: reduce single-source bias.

### 4.4 Physics diagnostics (`models/market_physics.py`)
Computes:
- **energy** (normalized return magnitude),
- **entropy** (state uncertainty measure).

Purpose: detect unstable conditions where conviction should be reduced.

### 4.5 Risk sizing (`risk/sizing.py`)
Uses constrained fractional Kelly with safety caps based on:
- uncertainty,
- shock probability,
- drawdown/consecutive loss context.

Purpose: translate signal quality into practical, bounded exposure.

---

## 5) Telegram report design

The Telegram output is optimized for decision making:
- top snapshot (signal/confidence/posterior/weight),
- model-risk block (regime mix + uncertainty + energy/entropy),
- market + sentiment context,
- key factors (top 3),
- scenario playbook (base/upside/defensive),
- data-quality warnings when inputs are missing.

Implementation lives in `telegram_bot.py` under `_format_message()`.

---

## 6) Setup

### 6.1 Requirements
- Python 3.10+
- Internet access for market/news/Telegram APIs

Install dependencies (example):

```bash
pip install -r requirements.txt
```

If your repo does not yet include `requirements.txt`, install at minimum:

```bash
pip install pandas numpy requests yfinance feedparser
```

---

## 7) Configuration

Set runtime settings in `config.py` (or via environment variables if your setup supports it), especially:

- Telegram:
  - `TELEGRAM_TOKEN`
  - `TELEGRAM_CHAT_ID`
- Signal thresholds:
  - `BULLISH_THRESHOLD`
  - `BEARISH_THRESHOLD`
- Weights:
  - `WEIGHT_SENTIMENT`
  - `WEIGHT_TREND`
  - `WEIGHT_VOLATILITY`
- Fusion/model tuning:
  - fusion component weights
  - uncertainty thresholds
  - shock/energy/entropy penalties
- RSS freshness policy:
  - `ALLOW_UNDATED_ARTICLES`

---

## 8) Run instructions

Run daily pipeline:

```bash
python main.py
```

Quick signal-engine sanity run:

```bash
python signal_engine.py
```

Compile check:

```bash
python -m py_compile $(rg --files -g '*.py')
```

---

## 9) Prediction history and validation

`predictions.csv` stores:
- predicted direction,
- confidence and posterior,
- feature/model diagnostics,
- recommended weight,
- outcome fields (`PENDING` then resolved).

`validator.update_pending_outcomes()` resolves pending rows against the next available trading close where possible.

Use this file for:
- rolling hit-rate,
- calibration diagnostics,
- drawdown/error analysis,
- model comparison over time.

---

## 10) Failure modes and safeguards

- Missing market schema -> reject frame early.
- Missing sentiment/market input -> confidence penalties + explicit data note.
- Undated RSS entries -> filtered by default unless explicitly allowed.
- High shock/energy/entropy -> confidence/weight throttling.
- Any critical signal engine exception -> safe neutral fallback payload.

---

## 11) Operational recommendations

1. Run once daily at consistent post-close time.
2. Monitor `predictions.csv` weekly for calibration drift.
3. Keep thresholds conservative until enough live samples accumulate.
4. Treat recommended weight as a **risk suggestion**, not auto-execution.
5. Recalibrate periodically after enough resolved outcomes.

---

## 12) Disclaimer

This system is for research and internal decision support.
It is **not** investment advice, and no component guarantees future returns.

