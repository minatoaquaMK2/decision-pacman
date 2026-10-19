# RFC: Architecture and key decisions

## Summary

The game is a browser application. A deterministic TypeScript engine steps at a fixed 30 ticks per second (30 Hz). An asynchronous agent loop, decoupled from the simulation, snapshots the state at upcoming junctions, encodes it as compact JSON (`features`), and queries a decision endpoint. By default the simulation never waits for the model. An optional Wait switch (`?wait=1`) holds the whole game at a junction until a slow model answers, so its choices are judged rather than its speed. The answer becomes a buffered direction intent that the engine applies at the next legal tile. There is no application server. The dev server forwards `/decide` to `VITE_DECISION_BASE_URL` (default `http://localhost:11434`), so the browser avoids CORS.

```
                    browser tab
 ┌──────────────────────────────────────────────────────┐
 │  requestAnimationFrame ─► Renderer (canvas) ◄─┐       │
 │                                               │ reads │
 │  30 Hz fixed-step ─► Engine.step(state, intent)       │
 │                           ▲           │ snapshot      │
 │                    intent │           ▼               │
 │                      Agent loop ─► Encoder ─► JSON    │
 │                           ▲                     │     │
 └───────────────────────────┼─────────────────────┼─────┘
                             │ answer + probs      │ POST /v1/systemone or POST /api/chat
                             └──── decision endpoint ◄┘
                                   (Ollama :11434, vLLM behind scripts/systemone_vllm.ts,
                                    or hosted Jev via dev-server proxy)
```

The system supports four kinds of players alongside scripted baselines (`random`, `greedy`):
1. Local decision models served over Ollama's `/v1/systemone` API (`tev1:0.8b`, `tev1:4b`, and fine-tuned models such as `pacman-0.8b-qwen`). The model returns probabilities for every legal option, read from option-letter logits. A fine-tuned model served by vLLM plays through `scripts/systemone_vllm.ts`, which serves the same endpoint in front of vLLM.
2. Plain chat models accessed through Ollama's `/api/chat` using policy names formatted as `llm:<ollama-model>` (e.g. `llm:phi4-mini`, `llm:gemma4:e4b`, `llm:qwen3.5:4b`). They receive the same state facts, a JSON schema restricting `move` to the legal options, thinking off, and temperature 0. Probabilities come from token logprobs.
3. TypeSafe's hosted decision model Jev (`jev-latest`) at `https://api.typesafe.ai/v1/systemone`, with a bearer key from `TYPESAFE_API_KEY`. The headless `jev` policy calls it directly. In the browser, the dev-server proxy adds the key, so it never reaches the page. Hosted requests leave out Ollama's `keep_alive` field, which the API rejects.
4. OpenAI's Decisions API (`POST https://api.openai.com/v1/decisions`, model `gpt-6-luna`), with a bearer key from `OPENAI_API_KEY`, under the policy name `openai:<model>`. Its request has a different shape from `/v1/systemone` but carries the same facts: the encoded state as `input`, and one `choice` question with the encoder's instructions and the legal options. It returns a probability for every option. In the browser, the dev server proxies `/openai` and adds the key.

The engine has no DOM dependency. The same code runs headless in Node for unit tests, benchmarks, teacher data generation, and seeded game evaluations. In the headless runner, a `realtime` clock converts decision latency into elapsed game ticks, while a `lockstep` clock pauses the game for every answer.

For training, the headless engine generates game states. Labels come from an OpenAI-compatible chat teacher (`teacher-peek5s`, Qwen3.8-27B on an RTX 5090) that sees one 5-second engine rollout per option at labeling time; the rollouts come from `src/agent/oracle.ts`. Lookahead may label training data but never enters a player's input in a comparison, so the student reads `features` only. The pipeline trains a LoRA adapter on the `Qwen/Qwen3.5-0.8B` base with unsloth, exports it to GGUF with llama.cpp, and imports it into Ollama, where it is served over `/v1/systemone` like any other decision model. The result, `pacman-0.8b-qwen`, is published as a Q8_0 GGUF with a Modelfile on Hugging Face (`grapeot/decision-pacman-0.8b-GGUF`).

```
 engine (Node) ──states──► teacher-peek5s (chat model + engine rollouts) ──labels──► dataset (JSONL)
 dataset (student reads features) ──► LoRA fine-tune (Qwen3.5-0.8B base, RTX 5090) ──► GGUF ──► /v1/systemone ──► evaluation
```

An iPhone app (`ios/`) hosts the web build inside a WKWebView and answers decisions natively (see decision 10). It supports three players: the fine-tuned 0.8B model running on device with llama.cpp on Metal, `phi4-mini` running on device with llama.cpp constrained by a GBNF grammar, and hosted Jev through a native HTTPS bridge that keeps the API key off the web page.

Current numbers and their caveats are in `results.md`. How to run, evaluate, and distill models is in `guides/`.

## Constraints from measurement

Full method and numbers are in `docs/model_evaluation.md`. The constraints that shaped this design:

1. **Latency is proportional to input tokens.** Decisions are prefill-bound: about 0.8 ms per input token for `nimble` on an M3 Ultra. Prefix caching saved only about 8% on a large state, and q4 quantization gave no speedup. The only effective lever is a smaller state.
2. **Representation matters more than model size.** On a full ASCII board, `nimble` was close to guessing (confidence 0.00–0.09). On per-direction facts, `tev1:4b` and `nimble` chose clear moves with probability around 0.99. A plain-English rendering of the same facts did not help.
3. **Off-the-shelf model choice.** `tev1:4b` was the best trade-off: about 160 ms and 0.88 on the probe. `tev1:0.8b` ran at about 65 ms but scored 0.40, near chance. `nimble` was slower (270 ms) and less accurate (0.68).
4. **The browser can call Ollama directly.** Ollama echoes `Access-Control-Allow-Origin` for `http://localhost:*` origins.
5. **Qwen3.8-27B works as a teacher.** Thinking off: 270 ms, 100% parseable, 0.97 on the probe, about 16 labels per second at concurrency 8. Thinking on: 1.1 s and 1.00. The serving stack does not support constrained decoding, so the teacher's answers are parsed from plain text.

## Components

- `src/engine/`: maze geometry, tile movement, ghost AI, pellets, scoring, lives, and level state. It exposes `createGame(seed, options)` and `step(state, intent)` as pure functions over plain data, driven by a seeded RNG. There are no timers and no DOM access. State events are written to an array on the state object.
- `src/encoders/`: transforms engine state and legal moves into model inputs via `(state, decisionPoint) => { state, instructions, options }`. Encoders compute facts with BFS over the maze and never call a model. Includes `features` (default compact JSON facts), board renderings (`ascii-window`, `ascii-full`), and `src/encoders/peek.ts` (`features-peek5s`, a lookahead encoder used as a labeling and record tool).
- `src/agent/`: the agent loop, decision logging, and policy implementations.
  - `src/agent/client.ts`, `src/agent/factory.ts`: client for `/v1/systemone` endpoints. Supports local Ollama and hosted Jev (`jev-latest`, bearer key in `TYPESAFE_API_KEY`, omits `keep_alive`).
  - `src/agent/decisions.ts`: `openai:<model>` policy for OpenAI's Decisions API (`/v1/decisions`, GPT-6 Luna). A refusal or a choice outside the legal options fails the decision.
  - `src/agent/llm.ts`: `llm:<model>` policy querying plain chat models via Ollama's `/api/chat` with a JSON schema restricting `move` to legal options, thinking off, and temperature 0.
  - `src/agent/prompt.ts`: the Ollama System One prompt in TypeScript, byte-identical to the Hugging Face chat template on 20 test fixtures, for models run outside Ollama (the iOS app).
  - `src/agent/native.ts`: the bridge policy for the iOS app: the page renders the prompt and native code runs the model.
  - `src/agent/oracle.ts`: engine rollouts. `rollout()` copies the game, plays an option, and runs the real engine forward with greedy play at later junctions; `peek()` reports a rollout as facts for the teacher's labeling prompt and the `features-peek5s` encoder. The same file defines a search player over rollouts (`oracle-5s`, `oracle-3s`, `oracle-8s`) that drives games during data generation; it sees the future, so it is never a compared player (README FAQ).
  - `src/agent/teacher.ts`: the chat-model teacher (`teacher`, `teacher-think`, `teacher-peek5s`) over any OpenAI-compatible endpoint.
  - `src/agent/players.ts`: the iOS app's three player ids, shared with the Swift side.
- `src/sim/runner.ts`: shared game-driving simulation loop used by the headless runner and dataset generation.
- `src/render/canvas.ts`, `src/main.ts`: Canvas 2D drawing, the HUD, controls, and the error panel. They read engine state and never change it.
- `src/audio/`: original chiptune music and sound effects synthesized in code using Web Audio. `npm run render-audio` renders an offline audio tour.
- `scripts/`: command-line entrypoints for headless execution, data generation, benchmarks, and probes:
  - `scripts/run_headless.ts`: headless game runner supporting realtime and lockstep clocks (`npm run headless`).
  - `scripts/gen_teacher_data.ts`: records game states labeled by the teacher (`npm run gen-data`).
  - `scripts/systemone_vllm.ts`: serves `/v1/systemone` in front of a vLLM server (`npm run systemone-vllm`). Each choice question becomes one `/v1/completions` request over the prompt from `src/agent/prompt.ts`, with one output token restricted to the option letters; the letters' log-probabilities, renormalized, are the answer.
  - `scripts/bench_prefill_decode.py`: measures latency split into prefill and decode across Ollama and llama.cpp backends.
  - `scripts/eval_general_decisions.py`: scores decision models on JevBench's public items, decisions that are not Pac-Man.
  - `scripts/probe_decision_models.py`, `scripts/probe_teacher.py`: probe a decision model or a chat teacher on 40 synthetic scenarios.
  - `scripts/record_demo.ts`: records a browser game to MP4 (`npm run record`).
  - `scripts/export_chat_prompts.ts`: exports game prompts with Ollama's recorded `llm:phi4-mini` answers for the iOS Mac check.
  - Dev and test entrypoints: `npm run dev`, `npm test`, `npm run typecheck`, `npm run build`.
- `training/`: Python scripts (using uv) for dataset preparation, fine-tuning, and evaluation:
  - `training/build_sft.py`: converts raw game state recordings into SFT datasets with smoothed hard teacher labels (or soft targets from per-option rollout values).
  - `training/train.py`: fine-tunes `Qwen/Qwen3.5-0.8B` with LoRA (unsloth), with the loss on the option-letter logits as Ollama scores them.
  - `training/export_gguf.sh`: merges LoRA adapters and exports to GGUF format via llama.cpp.
  - `training/prompt.py`: canonical Python prompt rendering matching Ollama's `/v1/systemone` format.
  - `training/score_ollama.py`: evaluates offline agreement on held-out validation states, supporting decision models, chat models (`--chat`), and hosted Jev.
  - `training/score_hf.py`: checks parity between Hugging Face model outputs and Ollama serving.
  - `training/compare_scores.py`: joins `score_hf.py` output with a served model's `score_ollama.py --out` file per state and reports matching choices, probability differences, and the margins where choices differ.
- `ios/`: SwiftUI application hosting the web build in a WKWebView with native decision handling (see decision 10). Runs the fine-tuned 0.8B model and `phi4-mini` on device via llama.cpp on Metal, and routes Jev requests through a native HTTPS bridge. Model files, API keys, the llama.cpp xcframework, and the web build are not committed to the repository.

## Key decisions

### 1. Pure frontend, browser calls the model endpoint directly

The browser fetches `<base>/v1/systemone` itself. A server would add a process and a network hop and buy nothing, because the model is already a local HTTP service. The base URL comes from `VITE_DECISION_BASE_URL` and can be changed in the UI. In development, Vite proxies `/decide/*` to the configured endpoint, which sidesteps CORS for other ports and origins. For a static build on another origin, the viewer adds that origin to `OLLAMA_ORIGINS`. The error panel detects each failure and shows the fix.

### 2. Fixed 30 Hz simulation, decoupled from rendering and from the model

The engine advances in fixed 1/30 s steps through an accumulator driven by `requestAnimationFrame`. Rendering draws the latest state every animation frame. The agent loop runs on its own async schedule. Nothing in the simulation awaits a network call, so a slow or failed request only means Pac-Man keeps following the last intent. Fixed steps keep the engine deterministic, which makes replays, tests, and headless runs reproducible.

### 3. Answers become planned turns or an immediate reversal

An exit answer becomes a turn planned for that junction tile (`StepInput.turn`), which the engine takes when Pac-Man reaches the tile center. A "back" answer turns Pac-Man around at once (`StepInput.reverse`). The two are kept apart on purpose. After a corner, a junction exit can point opposite to the current heading, and treating it as a reversal made Pac-Man dither. Keyboard play uses an untargeted buffered direction (`StepInput.intent`) with arcade semantics. Between decisions Pac-Man keeps going and follows single-exit corners, so a stale or missing answer costs a missed turn, never a stop in a corridor.

### 4. Decisions are about the next junction, with latency-sized lookahead

Real choices happen at junctions. Each request asks about the next junction on the current heading, with options set to that junction's exits plus "back". The agent keeps a moving average of decision latency and turns it into a lookahead distance (tiles Pac-Man covers while an answer is in flight, plus a margin). Suppose Pac-Man is closer to the next junction than that distance, and a turn there is already planned. Then the agent asks about the junction after it, because a new answer for the near one would arrive too late. An answer is discarded if its junction is no longer ahead. The HUD reports the stale rate. With lookahead, tev1:4b's stale rate fell from 22% to 10%.

### 5. Encoders report facts, never verdicts

An encoder that says "left is dangerous" turns the model into a rubber stamp on a scripted policy. An encoder that sends the raw board makes the model guess. The default `features` encoder sends, for each option, the BFS distance to the nearest ghost through that exit and whether it is frightened or approaching, pellets within N steps, distance to the nearest power pellet, and whether the path dead-ends within N steps. Global facts are frightened time left, pellets left, and lives. It never sends a safety label, a recommended move, or a per-option score. Game rules go into `instructions`. An `ascii-window` encoder (a crop around Pac-Man) and an `ascii-full` encoder exist for comparison and can be switched live.

This rule matters even more for fine-tuning. If the features contained verdicts, the fine-tuned model would learn to copy them, and the improvement would say nothing about the model.

### 6. The token budget is the latency budget

The `features` encoder targets about 200 tokens: short keys, no pretty-printing. Every encoder change records its input token count next to its score.

### 7. Models

The default is `tev1:4b` on Ollama. `tev1:0.8b` is the fine-tuning baseline. `nimble` and ollaya models remain selectable. The app sends `keep_alive: -1` during a game and one warm-up request at start.

### 8. Stack: Vite + TypeScript + Canvas 2D, no UI framework

The game is a canvas plus a small HUD, so a UI framework would add a second render cycle for no benefit. Vite provides the dev server, the proxy, and the static build. Vitest runs engine and encoder tests. The headless runner and data scripts are TypeScript run with `tsx`, importing the same engine and encoders as the browser. Training is Python, because the fine-tuning stack is.

### 9. Engine fidelity: faithful in behavior, not in cycles

The engine uses the classic 28×31 tile layout, sub-tile movement, the four ghost targeting rules, timed scatter and chase phases, frightened mode with seeded random turns, the ghost house, and the tunnel. Speeds are level 1 arcade values in tiles per second, scaled by a speed multiplier. Frame-exact speed tables and arcade bugs are out of scope. Graphics are drawn in code.

### 10. iPhone demo: one page, three players behind native bridges

The app runs the same web game and the same prompts as the browser. The page renders each prompt and native code runs it, so prompt text has one implementation (TypeScript, unit-tested) and the Swift side stays small. The fine-tuned 0.8B is scored from option-letter logits, as Ollama's `/v1/systemone` does. phi4-mini is asked as `llm:phi4-mini` asks Ollama, with a GBNF grammar in place of Ollama's JSON schema, and checked against Ollama's recorded answers on the Mac. Jev goes through native code because the page's `app://local` origin cannot call the API directly and the key must stay out of the page; the endpoint is fixed in Swift so the page cannot send the key elsewhere. Only one on-device model is loaded at a time. Model files and the key are copied into the app's Documents, never bundled.

## Headless runner

`scripts/run_headless.ts` plays N seeded games with any policy and writes one JSONL record per decision plus a per-game summary (score, pellets eaten, ticks survived, deaths, decisions, latency percentiles). Two clocks:

- `lockstep`: the simulation waits for each decision. It measures decision quality without latency and is how the teacher plays.
- `realtime`: each decision's measured latency converts to simulation ticks, which advance before the answer applies. It reproduces browser timing without rendering and is the clock for evaluation.

Baselines: `random` (uniform over legal options) and `greedy` (nearest pellet, avoid non-frightened ghosts within k steps).

## Stage 2: teacher data

**Labeler.** Qwen3.8-27B with lookahead (`teacher-peek5s`, `src/agent/teacher.ts`). For each option, `src/agent/oracle.ts` copies the game, plays the option, and rolls the real engine forward 5 seconds with greedy play at later junctions; the outcome (seconds until death, pellets, power pellets, ghosts eaten, points) goes into the teacher's prompt as facts. Ghost moves follow the real rules, so the lookahead is exact except for frightened ghosts' random turns. Without it, the same model lost all its lives within 39 to 93 s and was not a usable teacher on the encoded state alone (`docs/model_evaluation.md`, sections 9, 10, and 13). With it, the teacher labeled the `q1` dataset (54,325 requests, 31,570 deduplicated training rows). The teacher server returns no logprobs, so labels are hard.

**History.** The first student, `pacman-0.8b`, learned from the rollout search `oracle-5s` instead: it scores each option from the same rollouts (score gained, minus a death penalty that shrinks the later the death happens, minus the distance to the nearest pellet at the end) at about 5 ms of CPU per state. The search's per-option values allow soft targets, so near-ties do not force an arbitrary hard label. Both students play at the same level (`docs/model_evaluation.md` section 18). The LLM teacher is the main path because most tasks have no simulator to search with.

**State sampling.** If only a strong player drives the game, the data covers only the positions a strong player reaches, and the student will wander into positions it never saw. States are therefore sampled from a mix of players: the `oracle-5s` search, greedy, and random play. A DAgger round with the student playing was planned and not run. The teacher labels every state, whoever produced it.

**Information gap.** The teacher sees the rollouts, and the student sees only the encoding. Stage 4 measures agreement with the teacher on held-out states. A large gap points to the encoder, not the model.

**Record format.** One JSONL line per state: seed, tick, source policy, encoder, encoded state, instructions, options, per-option values, and the label.

**Splits.** Train, validation, and evaluation use disjoint seed ranges: evaluation 100-199, validation 900-999, training 1000 and up. `scripts/gen_teacher_data.ts` refuses evaluation seeds.

## Stage 3: fine-tuning

**Prompt parity with Ollama.** Ollama builds the scoring prompt in `decision/systemone.go`. It sends one user message, `{"context": <state>, "schema": [{"name", "description", "choices": [{"code": "A", "value", "description"}, ...]}]}` followed by `\n\nRequested field: "move"`, after the model's system prompt. The chat template is applied with thinking off. It then reads the logits of the option letters at the next position and takes a softmax over those letters only. `training/prompt.py` reproduces this byte for byte, including Go's HTML-safe JSON escapes and Tev1's system prompt. On 50 validation states, the served Q8_0 model and the bf16 training weights agree to within 0.027 in probability (median 0.010), and pick the same option on 48 of 50. The two that differ are near-ties.

**Base model.** `Qwen/Qwen3.5-0.8B`, the base of `tev1:0.8b` (Ollama reports 752M parameters). Tev1's own 0.8B weights are published only as GGUF.

**Data.** `training/build_sft.py` turns labeled states into rows with the Ollama messages and a target: the teacher's move with 0.9, and 0.1 spread over the other options (`--target label --smoothing 0.1`). For search labels it can instead take a softmax of each option's value divided by a temperature of 50 points (about five pellets). Exact duplicate states are removed. `q1` data: 31,570 training rows and 4,973 validation rows from separate seeds.

**Training.** `training/train.py`: LoRA (rank 16, alpha 32) on every language-side linear layer, lr 1e-4 cosine with 3% warmup, batch 32, 2 epochs (1,974 steps), bf16, one RTX 5090. The loss is cross-entropy between the target distribution and the softmax over option-letter logits at the answer position. Only that position is projected through the output layer, so memory stays small despite the 248k-token vocabulary. A startup check compares that projection with the model's own logits. Training takes about 32 minutes.

**Serving.** `training/export_gguf.sh` merges the adapter and converts to Q8_0 GGUF with llama.cpp. The GGUF is imported into Ollama with Tev1's system prompt, as `pacman-0.8b-qwen`, either through `/api/create` or through the Modelfile published with the weights on Hugging Face. Ollama 0.35.0 does not accept `CAPABILITY` in a Modelfile, but it serves `/v1/systemone` for the imported model anyway. The game needs no change beyond the model name.

The merged bf16 weights can also be served by vLLM, with no GGUF step. vLLM has no `/v1/systemone`, so `scripts/systemone_vllm.ts` serves it: one adapter in front of vLLM rather than a vLLM path in each client, so the game, the headless runner, and the scoring scripts run unchanged. It restricts the one output token to the option letters with `allowed_token_ids`, and vLLM must run with `--logprobs-mode processed_logprobs`, so the log-probabilities come after that restriction and include every option letter whatever its rank in the full vocabulary. Renormalized over the letters, they are the same softmax Ollama computes and the loss trains. By default vLLM returns the top tokens of the whole vocabulary; the fine-tuned model puts mass on letters past the last option, which pushed real options out of the list, so the adapter refuses to answer when a letter is missing.

**GPU use.** The GPU host serves the teacher model on all three cards. Training borrows one card with the cluster's lease script, which drops that replica from the router, and then returns it. The other two replicas keep serving.

## Stage 4: evaluation

Every model plays the same held-out seeds in the `realtime` clock at the same game speed on the same machine: `tev1:4b`, `tev1:0.8b`, the fine-tuned 0.8B, the teacher (lockstep, as an upper bound), `greedy`, and `random`. The metrics are pellets eaten, score, ticks survived, deaths, decision latency, and agreement with the teacher on held-out states. Results and plots go to `docs/model_evaluation.md`.

## Risks

- **Imported model not accepted by Ollama's decision runner.** Mitigation: the fallbacks in Stage 3. This is verified early with an untrained export, before any real training run.
- **Teacher labels are weaker on real positions than on the probe.** The probe scenarios have one clearly correct answer. Real positions are ambiguous. Mitigation: the lookahead in the teacher's prompt, and a check of its labels against the rollout search's choices (they agreed on 88-92% of states).
- **Distribution shift.** The student reaches states the teacher never did. Mitigation: mixed-policy sampling and a DAgger round.
- **Shared GPU host.** Teacher serving and training compete for the same GPU. Mitigation: run the stages in sequence.

## Error handling

The client distinguishes connection refused, CORS rejection, model not found, and 400/413 responses, and shows a specific fix for each. On any failure the game keeps running on the last intent, and the agent retries with backoff.

## Alternatives considered

- **Server-side game with a WebSocket viewer.** It adds a runtime and a protocol for a single-player demo whose model is already a local HTTP service.
- **Turn-based play in the browser.** It scores higher but removes the point of the demo. It survives as the `lockstep` clock.
- **Chat model as the in-game player.** Qwen3.8-27B is accurate but needs parsing and a remote GPU, and gives no calibrated probabilities. It is used offline as the teacher instead.
- **Labels from the rollout search alone.** They are free and exact for the rollout horizon, and the first student used them; it played at the same level as the LLM-distilled one. The LLM teacher is the main path because the recipe carries over to tasks that have no simulator.
