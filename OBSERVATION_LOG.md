# Coin MVP Observation Log

## 2026-05-10 Strategy Reset Baseline

Source: user report screenshot after four-day cloud paper run.

- Equity: 951,053 KRW
- Cumulative return: -4.9%
- PnL from 1,000,000 KRW: -48,947 KRW
- Completed trades: 74
- Win rate: 27.0%
- Payoff ratio: 0.69x
- Expectancy: -553 KRW
- Max drawdown: -40,931 KRW
- Open positions: 0
- Cash: 951,053 KRW
- Entries today: 6
- Consecutive losses: 2

Observed problems:

- Entry activity stopped during some periods because blocked hours and low daily entry caps interacted with strict filters.
- Strategy remained structurally unprofitable: payoff ratio below 1.0 and negative expectancy.
- Trade size was too small relative to fees, slippage, and stop size, so realized PnL was muted and often fee-sensitive.

Applied tuning direction:

- Removed blocked entry hours.
- Raised daily entry allowance while keeping max one new entry per tick.
- Added minimum trade cash threshold.
- Kept reward-risk and crash-candle filters.
- Opened Bollinger confirmation from 2 confirmations back to 1 to avoid over-filtering.
- Widened stop from 0.65% to 0.85%, raised final take profit to 2.0%, and moved partial take profit to 1.2%.

Next evaluation thresholds:

- Minimum observation: 24 hours or 20 completed trades, whichever comes later.
- First meaningful read: 30 completed trades.
- Tuning-quality read: 60 completed trades.
- Strategy-confidence read: 100 completed trades.

Decision rules:

- If payoff ratio stays below 1.0 after 30 completed trades, do not increase exposure.
- If expectancy remains negative after 60 completed trades, change exit logic before changing entry logic.
- If max drawdown exceeds -7% before 60 completed trades, reduce position fraction and daily entry count.
- If win rate is below 35% but payoff ratio is above 1.5, continue observing.
- If win rate is below 35% and payoff ratio is below 1.0, pause and redesign.
- If payoff ratio is above 1.2 and expectancy is positive after 60+ completed trades, continue the same settings for another day.

## Planned Observation Window

- Start: after 2026-05-10 cloud deployment of the tuned config.
- Review window: 2026-05-13 to 2026-05-14 KST.
- Reason: one day may be too short because strict filters can produce too few completed trades.
- Preferred sample target: at least 30 completed trades on the new settings, 60 if the market is active enough.
- Do not tune early unless drawdown approaches -7% or the bot clearly stops making valid scans.

## 2026-05-12 Market-Mode Strategy Update

Source: user report screenshot after continued cloud paper run.

- Equity: 950,954 KRW
- Cumulative return: -4.9%
- Completed trades: 81
- Win rate: 28.4%
- Payoff ratio: 0.74x
- Expectancy: -499 KRW
- Entries today: 0 at the time of review

Observed problems:

- Entry logic still became inactive during weak or quiet market periods.
- BTC/context filters were too binary: weak BTC often meant no candidates, even when individual coins had rebound setups.
- The dashboard did not clearly show the current operating regime, making it hard to know whether the bot was defensive or active.

Applied tuning direction:

- Replaced the binary BTC/context gate with a graded market mode: `risk_on`, `neutral`, `risk_off`, `capital_protect`.
- Added KST session labels so the report shows whether the bot is in morning, midday, evening liquidity, US overlap, or late quiet mode.
- Risk-on mode can use up to two new entries per tick; neutral/risk-off stays limited to one new entry.
- Position sizing is scaled by market mode: larger in risk-on, normal in neutral, smaller in risk-off, blocked only in capital-protect.
- Relaxed several entry constraints to be more progressive: volume ratio, orderbook imbalance, market breadth, BTC momentum, reward-risk ratio, and daily entry count.

Evaluation rules for this update:

- First check after upload: report must show `시장 모드` and `시장 세션`.
- If the bot logs repeated `capital_protect`, confirm BTC/global market is actually weak before changing thresholds again.
- If it reaches 30 completed trades and expectancy remains below 0, adjust exits before adding more entries.
- If it reaches 60 completed trades and payoff ratio remains below 1.0, redesign target/stop logic.
- If drawdown reaches -7% before 60 completed trades, reduce position fraction before changing entry filters.

## 2026-05-17 Conviction Sizing Reset

Source: user report screenshot after extended cloud paper run.

- Equity: 915,462 KRW
- Cumulative return: -8.5%
- Completed trades: 156
- Win rate: 35.9%
- Payoff ratio: 0.81x
- Expectancy: -443 KRW
- Max drawdown: -75,040 KRW
- Open positions: 0

Observed problems:

- Sample size is now large enough to conclude the previous strategy is structurally negative.
- Exit attribution shows positive partial exits, but stop-loss and trend-exit groups dominate total loss.
- Trade size around the 180k-220k KRW range was too small to make winning trades matter after fees, while losses still accumulated.
- The bot was active, but activity quality was not high enough.

Applied tuning direction:

- Move from many small exploratory entries to fewer conviction entries.
- Raise position fraction and minimum cash per trade so each valid entry is meaningful.
- Keep the absolute daily entry cap high and let situational controls decide whether entries are acceptable.
- Add a candidate-score floor so weak setups are skipped even when filters technically allow them.
- Raise the candidate-score floor dynamically when drawdown, consecutive losses, or heavy same-day activity appear.
- Reduce position size dynamically during drawdown or after consecutive losses instead of fully stopping by a fixed entry count.
- Add a validated-recovery gate so rebound entries require actual price recovery, stronger close position, and volume confirmation.
- Add a net-edge gate so expected upside must remain meaningfully positive after estimated downside, fees, and slippage.
- Penalize candidate scores when expected downside is high, instead of ranking only by confidence and upside.
- Tighten stop-loss and dynamic downside so bad entries are cut earlier.
- Let small trend-break losses breathe until stop-loss or time-stop, instead of closing every weak trend break immediately.
- Require stronger expected upside, volume, orderbook, and breadth before deploying larger capital.

Next evaluation thresholds:

- Do not judge before 20 completed exits on the new version.
- First review at 30 completed exits.
- Stronger read at 50 completed exits.
- If expectancy remains negative after 50 exits, change the entry thesis, not just sizing.
- If max drawdown reaches -3% from the new deployment equity before 30 exits, reduce size immediately.
