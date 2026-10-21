# Results

One run of the full benchmark (46 scenarios) with `Qwen/Qwen2.5-7B-Instruct`, greedy decoding, on a
Kaggle GPU. Raw per-scenario rows and full agent traces: `results/rows.json`, `results/traces.json`;
summary: `results/summary.json`. One model and one run: treat differences as indicative.

## Headline

| System | Correct | 95% CI (bootstrap) | Ran without error | Recovered from an error | Asked a human | LLM calls / scenario | Seconds / scenario |
|---|---|---|---|---|---|---|---|
| Single LLM, one shot | 54.3% (25/46) | 41.3 to 69.6 | 45.7% | 0 | 6 | 1.0 | 8.1 |
| Single LLM with retry loop | 82.6% (38/46) | 69.6 to 93.5 | 73.9% | 13 | 12 | 1.4 | 11.1 |
| Multi-agent (4 agents + tools) | 84.8% (39/46) | 73.9 to 93.5 | 78.3% | 15 | 5 | 5.1 | 35.3 |

Exact McNemar tests on correctness: multi-agent vs one shot p = 0.0001; retry loop vs one shot
p = 0.0002; **multi-agent vs single LLM with the same retry loop p = 1.0**.

## What this shows

1. **The error-feedback loop is what helps.** Feeding validator and simulator errors back to the
   model raised correctness from 54.3% to about 83 to 85%. Both loop-based systems recover from
   conflicting settings (8/8) that a one-shot answer never fixes (0/8).
2. **Splitting the work into four agents did not measurably beat one LLM with the same loop.**
   84.8% against 82.6% is one scenario (a single-cell case: 6/6 against 5/6), p = 1.0, at about 3.6
   times the LLM calls and 3.2 times the time. Any benefit of the roles is not detectable here.
3. **The Reviewer agent is noisy.** It objected in 30 of the 46 scenarios, often against the
   documented rules (for example demanding a 4 dB hysteresis for a single cell, which the simulator
   forbids), which burns attempts.
4. **No system solved the resource-limit scenarios (0/6 each).** A run that is too large fails only
   when the simulator starts. Under the original check a correct fix keeps the user count and
   shortens the run. The models mostly did the opposite or gave up: the retry loop asked a human in
   all 6, one multi-agent run completed but cut the user count, and the rest failed. See the caveat.
5. **Human involvement.** On the 6 genuinely vague requests every system asked a human in 5 of 6.
   The one-shot baseline also asked wrongly on one clear single-cell request.

## Per category (correct / total)

| Category | One shot | Retry loop | Multi-agent |
|---|---|---|---|
| explicit | 12/12 | 12/12 | 12/12 |
| derived (for example "pedestrian" = 3 km/h) | 8/8 | 8/8 | 8/8 |
| conflict (bandwidth not allowed on carrier) | 0/8 | 8/8 | 8/8 |
| single_cell (hysteresis with one cell) | 0/6 | 5/6 | 6/6 |
| resource (run too large) | 0/6 | 0/6 | 0/6 |
| ambiguous (should ask a human) | 5/6 | 5/6 | 5/6 |

## Caveats

- The simulator's resource error says the run is too large but not what to reduce, so the choice
  between cutting users and cutting time is genuinely ambiguous. The strict check (keep the users,
  shorten the time) was fixed before the run and is reported as is; it was not loosened afterwards.
  Treat the resource row as a benchmark-design finding more than a model failure.
- Small benchmark (46 scenarios), two phrasings per category, one 7B model, greedy decoding.
- The simulator is a teaching-grade model. "Correct" means matching the ground-truth configuration
  properties, not that the simulated physics is calibrated.
- Not tested here: a second model, a larger model, or a real third-party simulator.
