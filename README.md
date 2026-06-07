# Market Signal System

A free, weekly market-signal report pipeline for **NIFTY 50** that combines:
- market structure features,
- seven-day RSS/news sentiment,
- internal probabilistic models (state-space + regime + fusion),
- physics-inspired stability diagnostics,
- conservative risk sizing,
- biggest weekly movement attribution with associated news headlines,
- and Telegram delivery.

The project is designed to stay low-cost, interpretable, and operationally simple.

---

## 1) What this system does

Each weekly run produces one decision package:
- **Signal**: `BULLISH`, `BEARISH`, or `NEUTRAL`
- **Confidence**: bounded confidence score
- **Posterior probability**: fused `P(up)`
- **Diagnostics**: regime, uncertainty, disagreement, energy, entropy
- **Recommended weight**: constrained fractional Kelly output
- **Weekly movement/news context**: the largest NIFTY 50 moves of the week with same-day RSS headlines that likely coincided with, or contributed to, the move
- **Narrative**: top reasons and scenario playbook

Outputs are:
1. saved to `predictions.csv`,
2. logged for traceability,
3. sent to Telegram as a formatted weekly report.

> Note: the movement-news block is attribution by date, direction, and sentiment strength. It is intentionally phrased as associated/likely driver context, not proven causality.

---

## 2) Repository structure

```text
market-signal-system/
├── main.py                 # Orchestrates full weekly run
├── config.py               # Runtime configuration + validations
├── logger.py               # Logging setup
├── market_data.py          # OHLCV fetch + schema validation
├── news_fetcher.py         # RSS fetch + weekly freshness filtering
├── sentiment.py            # Sentiment scoring
├── weekly_report.py        # Biggest weekly moves + associated news attribution
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

Automation lives in `.github/workflows/weekly_report.yml` and runs once weekly after Friday's regular market close.

---

## 3) End-to-end flow

1. `market_data.fetch_market_data()` downloads NIFTY OHLCV, normalizes column names, validates schema.
2. `validator.update_pending_outcomes()` resolves any prior pending prediction rows when fresh close data is available.
3. `features.compute_features()` builds trend/momentum/volatility/RSI context.
4. `news_fetcher.fetch_news()` collects RSS items and keeps articles from the seven-day report window.
5. `sentiment.analyze_sentiment()` scores weekly news polarity.
6. `signal_engine.generate_signal()` computes:
   - base heuristic score,
   - latent state estimate,
   - regime probabilities,
   - physics diagnostics,
   - posterior fused probability,
   - confidence and recommended weight.
7. `weekly_report.build_weekly_movement_news()` finds the largest weekly close-to-close moves and attaches same-day, sentiment-ranked headlines.
8. `validator.save_prediction()` appends a row to `predictions.csv`.
9. `telegram_bot.send_weekly_report()` sends the decision report to Telegram.

---

## 4) Core quantitative components

### 4.1 State-space model (`models/state_space.py`)
Estimates latent market state:
- drift,
- acceleration,
- volatility proxy,
- short-horizon directional probability (`p_up_1d`).

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

The Telegram output is optimized for weekly decision making:
- top snapshot (signal/confidence/posterior/weight),
- model-risk block (regime mix + uncertainty + energy/entropy),
- market + weekly sentiment context,
- biggest weekly moves with associated news headlines,
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
pip install pandas numpy requests yfinance feedparser textblob
python -m textblob.download_corpora
```

---

## 7) Configuration

Set runtime settings in `config.py` (or via environment variables if your setup supports it), especially:

- Telegram:
  - `TELEGRAM_TOKEN`
  - `TELEGRAM_CHAT_ID`
- Report window:
  - `MAX_ARTICLE_AGE_HOURS` (default: `168` for seven days)
  - `REPORT_LOOKBACK_TRADING_DAYS` (default: `5`)
  - `TOP_MOVEMENT_NEWS` (default: `3`)
- Signal thresholds:
  - `BULLISH_THRESHOLD`
  - `BEARISH_THRESHOLD`
- Weights:
  - `WEIGHT_SENTIMENT`
  - `WEIGHT_TREND`
  - `WEIGHT_VOLATILITY`
- Fusion/model tuning:
  - fusion component weights,
  - uncertainty thresholds,
  - shock/energy/entropy penalties
- RSS freshness policy:
  - `ALLOW_UNDATED_ARTICLES`

---

## 8) Run instructions

Run weekly pipeline:

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
- No same-day headline for a major move -> report says attribution is unavailable rather than inventing causality.
- High shock/energy/entropy -> confidence/weight throttling.
- Any critical signal engine exception -> safe neutral fallback payload.

---

## 11) Operational recommendations

1. Run once weekly after the Friday close, or manually after a shortened/holiday week.
2. Monitor `predictions.csv` weekly for calibration drift.
3. Keep thresholds conservative until enough live samples accumulate.
4. Treat recommended weight as a **risk suggestion**, not auto-execution.
5. Recalibrate periodically after enough resolved outcomes.

---

## 12) Disclaimer

This system is for research and internal decision support.
It is **not** investment advice, and no component guarantees future returns.

## Advanced Internal Analysis (math + physics)

The system includes:
- Kalman state-space latent drift/acceleration estimation (`models/state_space.py`)
- Regime probability inference for diffusion/trend/shock (`models/regime_hmm.py`)
- Physics diagnostics: market energy + entropy (`models/market_physics.py`)
- Bayesian probability fusion (`models/fusion.py`)
- Constrained fractional Kelly sizing (`risk/sizing.py`)
