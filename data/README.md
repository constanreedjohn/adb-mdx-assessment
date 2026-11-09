# Simulated data

## The story

Each evening, interest-rate data lands in MDX and an analyst reviews the exception flags it raises. MDX uses one generic rule: flag anything that changed by more than a set percentage. On a busy day that means hundreds of flags, and percentages mislead in both directions:

- A tiny move on a near-zero rate looks huge.
- A large move on a high rate looks small.

The first thing we are building is a system that turns that list into one short answer: **what to look at first, and why.** The analyst still makes every decision.

## The problem being assessed

For each flag, the system answers one question: **is this percentage telling the truth about how much the move matters?** It does that by measuring the change in basis points and comparing it with the rate's own history.

| Problem                                    | What goes wrong today                                                                                   | What the system must conclude         |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| **False alarm from a near-zero rate**      | A move of a few bps on a rate close to zero shows as a huge percentage, so a harmless flag looks urgent | A threshold artifact, low severity    |
| **Real move hidden by a small percentage** | A large move on a high rate shows as a small percentage, so it is easy to skim past among louder flags  | A genuine large move, to review first |
| **Ordinary noise**                         | A normal move lands just over the rule and adds to the pile                                             | Ordinary, not raised as a concern     |

Before any of that, the system checks one more thing: did the expected file arrive complete? On the example day it did.

**Not assessed yet:**

- Whether a large move is a valid market move or a data error. The system says it is large and should be reviewed first, and the analyst decides.
- Materiality, priority by currency, tenor and liquidity, stale data, isolated spikes and volatility surfaces. These come later and will have their own simulated days.

## The example day: 15 October 2026

This is the day the walking skeleton runs on. It is a **happy case**: the file arrives complete and nothing fails during the run. The data still contains problems for the system to find.

MDX's rule (more than 2%) raises **28 flags** across 80 rates (8 currencies × 10 tenors):

| Case                                | Flags                        | MDX shows             | Real change                            | Correct result                         |
| ----------------------------------- | ---------------------------- | --------------------- | -------------------------------------- | -------------------------------------- |
| **False alarm** (ADB's own example) | JPY 3M, 0.04% → 0.07%        | +75.0%                | +3 bps, normal for this rate           | `threshold_artifact`, low severity     |
| **The real item**                   | GBP 2Y, 4.0227% → 4.2027%    | +4.5%                 | +18 bps, about 5× its usual daily move | `genuine_large_move`, **review first** |
| Other false alarms                  | JPY 1M, CHF 1M, CHF 6M       | +68.5%, −29.1%, +5.9% | Under 3 bps each                       | `threshold_artifact`                   |
| Ordinary noise                      | 23 flags, for example EUR 3M | Just over the rule    | Normal for that curve                  | `ordinary`                             |

**The expected outcome:** 28 alarming-looking flags become one thing to look at first, GBP 2Y, with the reason and the evidence behind it.

## The files in each day

| File                              | Simulates                                        | Contents                                                           | Used by                                                                                       |
| --------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `mdx_linear_rates_<yyyymmdd>.csv` | The vendor's rates file as loaded into MDX       | Today's 80 rates                                                   | Input checks                                                                                  |
| `file_metadata.json`              | MDX's record of the delivery                     | Arrival time, size, record count, content hash                     | Input Validation Agent: did the file arrive complete?                                         |
| `flags.json`                      | MDX's exception list                             | Each flag's previous and current value, unit, rule and colour code | Input Validation and Market Data Assessment Agents                                            |
| `curve_values.csv`                | MDX's stored history                             | Every point for 61 days (60 of history plus today)                 | Market Data Assessment Agent: how unusual is this move, and did neighbouring tenors move too? |
| `ground_truth.json`               | Nothing at ADB: **the answer key**               | The correct class for each flag and what to review first           | Tests and evaluation only                                                                     |
| `manifest.json`                   | Nothing at ADB: a record of how the day was made | Seed, generator version, config version, a hash of each file       | Reproducibility                                                                               |

## Rules

- **Agents must never read `ground_truth.json`.** The MDX tool doesn't serve it, and tests check that it can't leak. If an agent could see the answers, the results would prove nothing.
- **Results on simulated data show that the process works.** They are not evidence of accuracy on ADB's data, so don't report them as accuracy.
- **Nothing here is real, so don't treat it as market data** or use it outside this project.
