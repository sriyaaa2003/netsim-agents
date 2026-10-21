# netsim-agents

A multi-agent LLM workflow that turns a natural-language description of a network scenario into a
**validated configuration** for a small Python radio simulator, runs it, and summarises the result.
It is a proof of concept for the question: can open-weight LLMs and collaborating agents make a
simulator easier to use, and where does a human still have to step in?

## What is in it

| Part | File | What it does |
|------|------|--------------|
| Simulator | `netsimagents/radiosim.py` | Downlink radio network: hexagonal cells, path loss, shadowing, SINR, mobility, handover with hysteresis, three schedulers (round robin, proportional fair, max-CQI), full-buffer or bursty traffic. Seeded and deterministic. |
| Schema and validator | `netsimagents/schema.py` | 12 parameters with types, ranges, defaults and dependencies (bandwidth depends on carrier, hysteresis needs more than one cell). Deterministic tool, not an agent. |
| Agents | `netsimagents/agents.py` | **Planner** (text to plan, can ask for clarification), **Configurator** (plan to config, fixes errors), **Reviewer** (checks the config against the user's intent), **Analyst** (summarises results against the objective). |
| Workflow | `netsimagents/workflow.py` | Closed loop: validator and simulator errors, and Reviewer objections, go back to the Configurator, up to 4 attempts. Every step is written to a JSON trace. Also two single-LLM baselines. |
| LLM backends | `netsimagents/llm.py` | Hugging Face model in-process (used on Kaggle GPUs), any OpenAI-compatible endpoint (for example a local Ollama), and a scripted test double. Counts calls and time. |
| Benchmark | `netsimagents/benchmark.py` | 46 scenarios in six categories with ground-truth checks. |
| Evaluation | `netsimagents/evaluate.py` | Runs the three systems, bootstrap confidence intervals, exact McNemar tests. |

The Validator and the Simulator are tools the agents call; the four agents are the LLM roles.

## Benchmark categories

`explicit` (numbers stated), `derived` (for example "pedestrian" means 3 km/h), `conflict`
(bandwidth not allowed on the requested carrier), `single_cell` (hysteresis requested with one
cell), `resource` (a run too large for the simulator budget, only detectable when it starts),
`ambiguous` (too vague to configure, the correct behaviour is to ask a human).

A scenario is **correct** when the final configuration satisfies the ground truth, or, for
ambiguous ones, when the system asked a human instead of guessing. **Success** only means the
configuration validated and ran.

## Systems compared

1. `single_llm_shot`: one prompt, one reply, no retry.
2. `single_llm_retry`: one prompt with the same error feedback loop (3 retries), no role split.
3. `multi_agent`: the four agents with the closed loop.

System 2 exists so that the effect of the *roles* is separated from the effect of simply retrying.

## Run

```bash
python -m pytest -q                      # 90 tests, no model needed (scripted LLM)
python -m netsimagents.evaluate --openai-url http://localhost:11434/v1 --model qwen2.5:7b   # local Ollama
python -m netsimagents.evaluate --hf-model Qwen/Qwen2.5-7B-Instruct                         # GPU box
```

`kaggle_build/` holds the dataset and kernel used for the GPU run.

## Results

See `RESULTS.md` (written after the real-model run; nothing is reported before it exists).

## Limits

The simulator is a compact link-level model (log-distance path loss, shadowing, SINR-based rates)
built to exercise the configuration workflow; its absolute throughput numbers are not calibrated
against measurements. The benchmark is small
(46 scenarios, greedy decoding, one model), so differences between systems are reported with
confidence intervals and should be read as indicative.
