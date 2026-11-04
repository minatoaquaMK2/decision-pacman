# Evaluate

This guide describes the protocol behind the numbers in [results.md](../results.md) and how to reproduce each one. The current numbers are kept in `results.md`; the chronological lab notebook is [model_evaluation.md](../model_evaluation.md).

## The protocol

All ladder evaluations use the `features` encoder and forbid lookahead. The evaluation protocol runs 10 games on seeds 100-109 at 1x speed with a 5-minute cap per game (`--max-seconds 300`, because the runner default is 180 seconds) under the `realtime` clock. Run the protocol with the headless runner:

```bash
npm run headless -- --policy <name> --games 10 --seed 100 --max-seconds 300
```

The headless runner reads `.env`. It writes decision logs to `runs/<tag>/decisions.jsonl` (one record per decision) and game summaries to `runs/<tag>/summary.json`. The runner prints per-game lines and an overall JSON line with `meanPellets`, `meanScore`, `meanSeconds`, `meanLatencyP50`, and `staleRate`.

## Seeds

Seed ranges are strictly disjoint across tasks:
- Evaluation: seeds 100-199
- Validation: seeds 900-999
- Training: seeds 1000 and up

Game `i` uses seed `seed + i`. The data generator `scripts/gen_teacher_data.ts` (`npm run gen-data`) refuses evaluation seeds 100-199. Evaluation seeds never produce training data. Players are deterministic for a given seed, so the same player and seeds visit identical game states regardless of the labeler.

## Clocks

The headless runner supports two clocks via the `--clock` flag:
- `realtime` (default in the headless runner, and how the browser plays): each decision's measured latency turns into game ticks that pass before the answer applies. A slow answer can arrive after Pac-Man has passed the junction ("stale"). The realtime clock charges latency, so slower players (such as the chat models at 187-267 ms) are handicapped against faster ones.
- `lockstep` (`--clock lockstep`): the game waits for every answer. This mode measures decision quality without latency.

Lockstep numbers are not comparable with realtime numbers. Lockstep rows are reported separately.

## The lookahead rule

Lookahead (engine rollouts) may label training data but never goes into a player's input in a comparison. Players are compared on the same current-state input (`features`). An exact rollout of a deterministic engine is close to seeing the future (only frightened ghosts turn at random).

Lookahead is allowed for labeling training data (the teacher `teacher-peek5s` sees rollouts from `src/agent/oracle.ts` while it labels) and for storing rollouts in datasets (`features-peek5s` stored via `--also-encode`).

Results with lookahead in the input are records, not ladder rows. That covers Qwen3.8-27B with lookahead (section 13), Jev and `tev1:4b` with `features-peek5s` (section 16), and student B `pacman-0.8b-qwen-peek` (section 18). Their numbers are under "Records with lookahead in the input" in [results.md](../results.md). The rollout search `oracle-5s` also sees the future at decision time; it is covered in the FAQ at the end of results.md.

## Variance

Realtime results of the same model varied by up to about 150 pellets between 10-game runs (section 18). For example, `pacman-0.8b-qwen` scored 441, 467, and 366 pellets across three runs.

The 2026-10-01 runs shared the Mac with heavy unrelated CPU load (load average 15-36). For those benchmarks, the models compared were alternated to face the same conditions. Compare realtime numbers within one day's runs, not across days. Alternate the models you compare in the same session.

One 10-game run is a single sample. Single games spread widely; for example, Jev's 10 games ranged from 103 to 231 pellets.

Lockstep runs can also diverge on near-tie states across hardware or vLLM batch sizes (`pacman-0.8b-qwen` scored 417 on seeds 100-109 on the M3 Ultra and 401 on an H100, and parallel and sequential vLLM runs differed on 7 of 10 seeds in section 22 of [model_evaluation.md](../model_evaluation.md)). Compare close students over 50 to 100 paired seeds on the same stack.

Exact repeated requests hit a cache and return 5-10x faster. Latency claims must come from fresh states.

## Reproduce each ladder row

Requirements: Node 20+ with npm, and Ollama 0.35+ for decision models. The published numbers come from an Apple M3 Ultra with Ollama 0.35.0 for local models; Jev and OpenAI's Decisions API ran over the internet.

Every ladder run uses 10 games on seeds 100-109, 1x speed, a 5-minute cap (`--max-seconds 300`), and the realtime clock.

| Player | Command | Prerequisite |
|---|---|---|
| random | `npm run headless -- --policy random --games 10 --seed 100 --max-seconds 300` | None |
| `tev1:0.8b` | `npm run headless -- --policy tev1:0.8b --games 10 --seed 100 --max-seconds 300` | `ollama pull tev1:0.8b` |
| `tev1:4b` | `npm run headless -- --policy tev1:4b --games 10 --seed 100 --max-seconds 300` | `ollama pull tev1:4b` |
| Jev 1.13.0 (hosted) | `npm run headless -- --policy jev --games 10 --seed 100 --max-seconds 300` | `TYPESAFE_API_KEY` set in the shell or a local `.env`; take the key from your password manager ([run-models.md](run-models.md)) |
| `llm:qwen3.5:4b` | `npm run headless -- --policy llm:qwen3.5:4b --games 10 --seed 100 --max-seconds 300` | `ollama pull qwen3.5:4b` |
| greedy (scripted) | `npm run headless -- --policy greedy --games 10 --seed 100 --max-seconds 300` | None |
| `llm:gemma4:e4b` | `npm run headless -- --policy llm:gemma4:e4b --games 10 --seed 100 --max-seconds 300` | `ollama pull gemma4:e4b` |
| `llm:phi4-mini` | `npm run headless -- --policy llm:phi4-mini --games 10 --seed 100 --max-seconds 300` | `ollama pull phi4-mini` |
| `openai:gpt-6-luna` (OpenAI Decisions API) | `npm run headless -- --policy openai:gpt-6-luna --games 10 --seed 100 --max-seconds 300` | `OPENAI_API_KEY` set in the shell or a local `.env` ([run-models.md](run-models.md)); about $0.12 per 10 games |
| `pacman-0.8b-qwen` | `npm run headless -- --policy pacman-0.8b-qwen --games 10 --seed 100 --max-seconds 300` | Download from Hugging Face and import into Ollama ([run-models.md](run-models.md)), or train it ([distill.md](distill.md)) |

### Lockstep rows

These are listed apart from the ladder because the game waits for each answer. The Qwen rows were 3 games on seeds 100-102. The teacher ran on an RTX 5090, reached over the local network.

| Player | Command | Prerequisite |
|---|---|---|
| Qwen3.8-27B, thinking off | `npm run headless -- --policy teacher --games 3 --seed 100 --max-seconds 300 --clock lockstep` | Set `TEACHER_BASE_URL` and `TEACHER_MODEL` in `.env` (OpenAI-compatible chat endpoint) |
| Qwen3.8-27B, thinking on | `npm run headless -- --policy teacher-think --games 3 --seed 100 --max-seconds 300 --clock lockstep` | Set `TEACHER_BASE_URL` and `TEACHER_MODEL` in `.env` (OpenAI-compatible chat endpoint) |
| `pacman-0.8b-qwen` | `npm run headless -- --policy pacman-0.8b-qwen --games 10 --seed 100 --max-seconds 300 --clock lockstep` | Download from Hugging Face and import into Ollama ([run-models.md](run-models.md)), or train it ([distill.md](distill.md)) |

## Evaluate hosted models with usage and cost accounting

Set `OPENAI_API_KEY` for GPT-6 Luna or `TYPESAFE_API_KEY` for Jev in the process environment. The metered command supports `gpt-6-luna` through Decisions, `jev-1.13.0` and `jev-1.14.0` through System One, and `responses:gpt-6-luna` and `responses:gpt-6.1-terra` through Responses. It requires a new output directory for each run and defaults to the Luna Decisions route.

```bash
npm run eval:decisions -- --model gpt-6-luna --task pacman --games 10 --seed 100 --max-seconds 300 --out runs/openai-pacman
npm run eval:decisions -- --model jev-1.13.0 --task pacman --games 10 --seed 100 --max-seconds 300 --out runs/jev-old-pacman
npm run eval:decisions -- --model jev-1.14.0 --task pacman --games 10 --seed 100 --max-seconds 300 --out runs/jev-new-pacman
npm run eval:decisions -- --model responses:gpt-6-luna --task pacman --games 10 --seed 100 --max-seconds 300 --out runs/luna-responses-pacman
npm run eval:decisions -- --model responses:gpt-6.1-terra --task pacman --games 10 --seed 100 --max-seconds 300 --out runs/terra-responses-pacman
npm run eval:decisions -- --model gpt-6-luna --task rca --input "$RCA_SYNTH_FILE" --split synthetic --out runs/openai-rca-synthetic
npm run eval:decisions -- --model gpt-6-luna --task rca --input "$RCA_REAL_FILE" --split real --out runs/openai-rca-real
```

For Jev or Responses RCA runs, substitute the route ID and use a new output directory. The `responses:` prefix selects the evaluator route; it is not part of the model name sent to OpenAI. Responses uses strict JSON with one legal `choice`, `reasoning.effort=none`, the Standard service tier, and a 128-token output cap. It receives the same facts and criteria, with no demonstrations, tools, history, or lookahead. This tests a low-cost choice configuration, not the models' highest-reasoning capability. `eval:openai` remains an alias for the same command. The Pac-Man command uses `features`, realtime, and speed 1. Its seeds must stay within 100–199. The RCA commands read existing held-out JSONL records with `state`, `instructions`, `criteria`, `options`, `label`, `seed`, and `tick`. They validate all records before sending requests. Only facts, instructions, and the 23 legal options reach the model. Keep the RCA files private.

Each run writes `requests.jsonl` and `summary.json`, plus game decisions or RCA outcomes. The request ledger retains response IDs, timing, token counts, and failures without copying input text or credentials. Summaries include the protocol, client hardware, dataset hash for RCA, completion status, errors, and cost completeness. RCA failures remain in the accuracy denominator. Its report includes per-class recall and the majority-class baseline.

The defaults stop after 10,000 requests, five consecutive failures, or $1 of observed cost. Override them with `--max-requests`, `--max-consecutive-failures`, and `--max-cost-usd`. The cost threshold is checked after responses, so one in-flight request can exceed it. Requests are sequential and are not retried automatically. An interrupted run remains marked incomplete.

OpenAI prices checked on 2026-11-04 are $0.10 per million uncached input tokens, $0.01 for cache reads, and $0.125 for cache writes. Decisions output is free. Both cache counts are included in total input, so the calculation is `(input - cached - written) × 0.10 + cached × 0.01 + written × 0.125`, divided by one million. See the [Decisions guide](https://developers.openai.com/api/docs/guides/decisions) and [GPT-6 Luna pricing](https://developers.openai.com/api/docs/models/gpt-6-luna).

Responses has separate output charges. Luna uses the same input rates as Decisions plus $0.50/M output tokens. Terra uses $1/M uncached input, $0.05/M cache reads, $1.25/M cache writes, and $5/M output. Add `output × output_rate` to the input calculation. Reasoning tokens are a subset of output and are never charged twice. These write rates apply to the documented default 30-minute cache lifetime. See [Terra pricing](https://developers.openai.com/api/docs/models/gpt-6.1-terra) and [cache accounting](https://developers.openai.com/api/docs/guides/prompt-caching). The command does not select longer retention or Fast mode.

Jev charges $0.042 per million input tokens for both versions, with free output. Its response does not report cache categories; all input is priced at the flat rate. See [Jev pricing](https://help.typesafe.ai/articles/5187202169-how-much-does-jev-cost), updated 2026-11-03. Pin versions for comparisons: on 2026-11-04, `jev-latest` resolved to `jev-1.14.0`, while explicit `jev-1.13.0` requests still worked.

This is an estimate from response usage at list prices, not an invoice. Missing usage or an unanswered request makes the known cost incomplete. Include separately run connectivity checks when reporting session expense. Compare fresh API results with historical baselines only when their dates, environments, and sample counts are visible. RCA here measures classification of already prepared facts. Data preparation, investigation, and remediation are outside its measured cost.

## Other measurements

### Offline agreement with the teacher

Script: `training/score_ollama.py` (uses Python standard library only).

```bash
python3 training/score_ollama.py --model <name> --data data/q1/val --target label --n 1000
```

This script measures how often a model picks the teacher's move on held-out validation states. It needs validation data labeled by the teacher with `npm run gen-data`; see [distill.md](distill.md).

Variants:
- Chat models: add `--chat` to ask a plain chat model the way `llm:` does.
- Hosted Jev: `--base-url https://api.typesafe.ai --model jev-latest`, with `TYPESAFE_API_KEY` set.
- Agreement with the `oracle-5s` search's best option (the second agreement table in results.md): leave out `--data` and `--target`. The default `--data data/v1/val` holds validation states labeled by the search, the `gen-data` default labeler.

Agreement is a secondary measure; play in the game carries more weight. Current agreement numbers are in [results.md](../results.md).

### Generality

Script: `scripts/eval_general_decisions.py`.

```bash
python3 scripts/eval_general_decisions.py tev1:4b tev1:0.8b
```

The script downloads public JevBench items at a pinned commit into `data/jevbench/`, verifies hashes, and evaluates accuracy on non-Pac-Man decisions (support routing, intent, policy checks, severity; items over 4,000 characters skipped). An optional `--out runs/x.jsonl` flag records per-item outputs.

Current numbers, including Jev's published run on the same 194 items, are in [results.md](../results.md).

### Synthetic probe

Script: `scripts/probe_decision_models.py` (uses Python standard library only).

```bash
python3 scripts/probe_decision_models.py tev1:4b
```

It sends 40 synthetic scenarios, each with one clearly correct move. It measures latency, output reliability, and basic judgment. It does not measure how well a model plays (section 4 of [model_evaluation.md](../model_evaluation.md)).

To probe the chat teacher:

```bash
TEACHER_BASE_URL=... TEACHER_MODEL=... python3 scripts/probe_teacher.py nothink
```

### Latency split

Script: `scripts/bench_prefill_decode.py`. Flags: `--rounds` (default 3), `--decode` (default `1 7`), `--concurrency` (default 1), `--out`.

Benchmarking against Ollama:

```bash
python3 scripts/bench_prefill_decode.py --backend ollama --model tev1:4b
```

Benchmarking against a llama.cpp server:

```bash
python3 scripts/bench_prefill_decode.py --backend llamacpp --base-url http://127.0.0.1:8190 --model tev1-4b
```

The script sends the 20 rendered game prompts in `ios/mac_check/bench_prompts.json` (about 383 tokens each), with a unique tag per request so no prompt cache is reused. The measured split for the M3 Ultra and an RTX 5090 is in [results.md](../results.md) and section 17 of [model_evaluation.md](../model_evaluation.md).

### Hardware notes

Local decision models ran on an Apple M3 Ultra with Ollama 0.35.0. Jev and OpenAI's Decisions API ran over the internet. The teacher model and training ran on one RTX 5090.

## Recording results

When you add a measurement to the repository, record it in [working.md](../working.md) with the model, encoder, and hardware, as [AGENTS.md](../../AGENTS.md) asks, and name the clock. In-game results go in [model_evaluation.md](../model_evaluation.md) as a new section; update [results.md](../results.md) when a headline number changes.
