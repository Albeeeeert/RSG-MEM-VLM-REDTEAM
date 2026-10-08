# RSG-Mem VLM Red-Teaming

RSG-Mem is a research prototype for **memory-augmented relational strategy graph
exploration** in automated vision-language-model red-teaming. It turns abstract
multimodal testing strategies into a persistent typed graph, retrieves relevant
cross-sample experience, and selects among four graph operations:

- `REUSE` an existing strategy;
- `REFINE` a strategy in response to an observed failure;
- `COMPOSE` complementary strategies through a multi-parent composition node;
- `INVENT` a new strategy when the current graph has a coverage gap.

The implementation is black-box and provider-neutral. It includes an entirely
offline mock backend for reproducible development and OpenAI-compatible adapters
for authorized evaluation endpoints.

## What is implemented

- persistent typed strategy graph with refinement, composition, similarity,
  complementarity, conflict, and subsumption relations;
- SQLite multimodal episodic memory;
- contextual retrieval over goal, image, cross-modal, refusal, target, and
  historical-utility signals;
- dual success/progress Beta posteriors and pairwise synergy posteriors;
- graph-guided candidate generation and Thompson-sampling selection;
- multimodal actuator with a constrained Pillow tool registry;
- pre-query strategy/image/text consistency checking and repair loop;
- multidimensional response evaluation and failure taxonomy;
- strategy/composition credit assignment and lifecycle management;
- cold-start, persistent online runs, JSONL traces, graph snapshots, and CLI;
- offline unit and end-to-end tests.

The full research-method specification is in [`docs/METHOD.md`](docs/METHOD.md).

## Architecture

```text
Sample context ──> multimodal memory retrieval ──> active strategy subgraph
                                                        │
                                                        ▼
                    REUSE / REFINE / COMPOSE / INVENT controller
                                                        │
                                                        ▼
                                  strategy designer + multimodal actuator
                                                        │
                                                        ▼
                                      consistency check / local repair
                                                        │
                                                        ▼
                                            target VLM + evaluator
                                                        │
                                                        ▼
                                 credit assignment + graph/memory update
                                                        └─────── loop
```

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### 2. Run the deterministic offline demo

```bash
rsg-mem demo --output runs/demo
```

Without installation:

```bash
PYTHONPATH=src python -m rsg_mem demo --output runs/demo
```

The demo writes:

```text
runs/demo/
├── graph.json
├── memory.sqlite3
├── run_summary.json
├── traces.jsonl
└── artifacts/
```

### 3. Run a configured experiment

```bash
cp configs/mock.yaml configs/local.yaml
PYTHONPATH=src python -m rsg_mem run \
  --config configs/local.yaml \
  --samples examples/samples.jsonl \
  --seeds examples/seeds.yaml
```

### 4. Inspect persistent state

```bash
PYTHONPATH=src python -m rsg_mem inspect --config configs/local.yaml
```

## OpenAI-compatible endpoints

Copy [`configs/openai_compatible.example.yaml`](configs/openai_compatible.example.yaml)
and set API keys through environment variables. Each role can point to a different
endpoint/model:

```yaml
providers:
  strategy_agent:
    type: openai_compatible
    base_url: http://127.0.0.1:8001/v1
    model: attacker-model
    api_key_env: ATTACKER_API_KEY
  target:
    type: openai_compatible
    base_url: http://127.0.0.1:8002/v1
    model: target-vlm
    api_key_env: TARGET_API_KEY
  judge:
    type: openai_compatible
    base_url: http://127.0.0.1:8003/v1
    model: evaluator-model
    api_key_env: JUDGE_API_KEY
```

The framework sends local images as base64 data URLs. Strategy generation,
realization, consistency checking, and judging use strict JSON schemas with
defensive parsing.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The end-to-end test uses only the deterministic mock backend and makes no network
requests.

## Repository layout

```text
src/rsg_mem/
├── actuator.py              # constrained multimodal execution
├── cli.py                   # command-line interface
├── config.py                # YAML configuration
├── controller.py            # graph-guided action generation/selection
├── credit.py                # node and relation credit assignment
├── embeddings.py            # deterministic local embeddings
├── engine.py                # complete three-loop workflow
├── graph.py                 # persistent typed strategy graph
├── lifecycle.py             # promotion/retirement/reactivation
├── memory.py                # SQLite experience/posterior memory
├── models.py                # shared typed records
└── providers/               # mock and OpenAI-compatible adapters
```

## Evaluation protocol

For defensible comparisons, keep target-query budgets, agent/judge/image calls,
models, tools, and seeds fixed across baselines. In particular, compare against a
TreeTeaming-style controller receiving the **same retrieved history** but without
typed graph relations. This separates relational reasoning gains from gains caused
only by additional context.

Use the framework only with controlled benchmark data and evaluation endpoints
you are authorized to test.
