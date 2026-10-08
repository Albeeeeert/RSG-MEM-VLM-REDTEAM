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
                    REUSE / REFINE / COMPOSE / INVENT controller + agent advisor
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

## How the agent participates

The default `configs/mock.yaml` uses a deterministic `RuleBasedStrategyAgent`; it does **not**
call an LLM. This keeps tests offline, but it can make the agent layer look invisible. With an
`openai_compatible` strategy provider, every search round performs the following agent loop:

1. the controller retrieves cross-sample memories and builds graph candidates;
2. the **action advisor** ranks `REUSE`, `REFINE`, `COMPOSE`, and `INVENT` candidates;
3. graph utility and agent advice are fused using `agent.advisor_weight`;
4. the **strategy designer** creates a refinement, composition, or new abstract strategy;
5. the **test-case realizer** converts it into an image-text instance;
6. the **consistency checker** can trigger the **repair agent** before a target query;
7. the **response judge** classifies feedback, which is written back to graph and memory.

The six editable prompts are versioned in
[`src/rsg_mem/prompt_templates/`](src/rsg_mem/prompt_templates/). Inspect them directly:

```bash
PYTHONPATH=src python -m rsg_mem prompts
PYTHONPATH=src python -m rsg_mem prompts --show action_advisor
PYTHONPATH=src python -m rsg_mem prompts --show strategy_designer
```

The advisor receives structured evidence rather than an unbounded transcript: candidate scores,
local graph nodes and relations, retrieved successes and failures, recent target feedback, failure
signature, stagnation count, and remaining query budget. Its output is validated against the
candidate IDs before it can affect selection. The fusion weight and evidence limits are configured
as follows:

```yaml
agent:
  enabled: true
  advisor_weight: 0.35
  memory_items: 5
  recent_attempts: 4
  graph_nodes: 12
```

To override prompts for an experiment, copy any or all six same-name Markdown files to another
directory and set `prompt_dir` in the YAML config. Missing override files fall back to built-ins:

```yaml
prompt_dir: ../prompt_overrides
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

The `strategy_agent` model performs action advising, abstract strategy design, concrete
test-case realization, and local repair. The framework sends local images as base64 data URLs;
agent, consistency, and judge outputs use structured JSON with defensive parsing.

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
├── prompting.py             # prompt loading and override support
├── prompt_templates/        # six versioned agent/evaluator prompts
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
