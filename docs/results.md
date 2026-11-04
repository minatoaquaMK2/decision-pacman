# Results

This document reports pellets eaten per game and decision latency across players. The full lab notebook is [model_evaluation.md](model_evaluation.md), whose section numbers are cited below.

## How to read these numbers

All ladder benchmarks use the `features` input without lookahead. Evaluations run across 10 games on seeds 100-109 at 1x speed with a 5-minute cap per game under the realtime clock. A single level contains 244 pellets. "Pellets" indicates the mean number of pellets eaten per game. Latency is the median decision time (p50).

The runtime supports two clocks:
- `realtime` (default): measured decision latency translates into game ticks that pass before an action applies. Late decisions arrive after Pac-Man has already passed a junction ("stale").
- `lockstep`: the simulation waits for every answer, measuring decision quality independent of latency.

Lockstep results are not comparable with realtime results. The same holds for browser play with the Wait switch on, which holds the game at a junction until the model answers.

The lookahead rule ([AGENTS.md](../AGENTS.md); conclusion 10 of [model_evaluation.md](model_evaluation.md)): lookahead (engine rollouts) may label training data but never enters a player's input in a comparison. Players are compared on the same current-state input (`features`). An exact rollout of a deterministic engine is close to seeing the future (only frightened ghosts turn at random). Results with lookahead in the input are records, not ladder rows.

Historical ladder hardware: local models ran on an Apple M3 Ultra with Ollama 0.35.0. Jev ran over the internet against TypeSafe's hosted API, and GPT-6 Luna against OpenAI's Decisions API. The original Qwen3.8-27B teacher and student fine-tuning ran on one NVIDIA RTX 5090. Later H100 experiments are documented separately.

## Hosted comparison on 2026-11-04

The five fresh treatments below use 10 realtime games each, seeds 100-109, current-state `features`, and a 300-second cap. Responses rows use ordinary LLM calls with reasoning off and strict choice JSON. GPT-6.1 Luna was absent from the account model list, so GPT-6 Luna is the fallback. Full methods, standard errors, RCA class breakdowns, and token accounting are in [section 25](model_evaluation.md#25-fresh-hosted-decisions-and-ordinary-llm-choices-2026-11-04).

| Model / API | Pellets mean (SE) | Range | Mean game p50 | Game API cost |
|---|---|---|---|---|
| GPT-6 Luna · Decisions | 235.9 (20.6) | 157-377 | 201.0 ms | $0.09796270 |
| jev-1.13.0 | 187.1 (10.6) | 142-234 | 116.4 ms | $0.09137176 |
| jev-1.14.0 | 204.2 (8.6) | 157-253 | 101.4 ms | $0.16556035 |
| GPT-6 Luna · Responses | 66.1 (13.2) | 7-136 | 6137.7 ms | $0.00293330 |
| GPT-6.1 Terra · Responses | 106.6 (7.3) | 86-165 | 1114.0 ms | $0.11655900 |

The historical student remains 425 pellets/30 games at 55 ms. The new hosted batches used a Linux client at different times, while historical results used other hardware/network paths. Keep that distinction when comparing rows.

| Model / API | Synthetic (333) | Real (48) | Synthetic / real p50 | Synthetic / real cost |
|---|---|---|---|---|
| GPT-6 Luna · Decisions | 283/333 (85.0%) | 25/48 (52.1%) | 203.8 / 215.0 ms | $0.03681360 / $0.00451020 |
| jev-1.13.0 | 279/333 (83.8%) | 20/48 (41.7%) | 116.6 / 115.7 ms | $0.02260234 / $0.00285016 |
| jev-1.14.0 | 284/333 (85.3%) | 21/48 (43.8%) | 110.6 / 112.4 ms | $0.02260234 / $0.00285016 |
| GPT-6 Luna · Responses | 278/333 (83.5%) | 25/48 (52.1%) | 997.4 / 983.9 ms | $0.04927297 / $0.00528585 |
| GPT-6.1 Terra · Responses | 282/333 (84.7%) | 31/48 (64.6%) | 1182.8 / 1124.6 ms | $0.49373850 / $0.05273850 |

The historical RCA student scored 313/333 and 11/48; DeepSeek scored 272/333 and 20/48 on those same cases. The real-set majority baseline is 34/48 (70.8%), with labels from one annotator. These measurements cover classification of already prepared facts, excluding LLM preprocessing and downstream investigation.

Session API expense, including connectivity checks, is $1.16785656 across 15,143 requests. It comes from recorded usage at list prices, not an invoice. The [bilingual website](../site/index.html) includes the per-task ledger totals and an illustrative full-workflow cost calculator. Its default example saves 4% overall when classification is 10% of the bill, its replacement costs 10% as much, and added preparation costs 5% of the original total. Those assumptions are not measured pipeline savings.

## The historical ladder

All rows: `features` input, no lookahead, 10 games on seeds 100-109, 1x speed, 5-minute cap, realtime clock.

| Player | Mean pellets | Mean survival | Decision p50 | Source |
|---|---|---|---|---|
| random | 30 | 37 s | 0 ms | section 11 |
| `tev1:0.8b` | 69 | 35 s | 62 ms | section 11 |
| `tev1:4b` | 148 | 55 s | 202 ms | section 11 |
| Jev 1.13.0 (hosted) | 178 | 52 s | 115 ms | sections 11, 12 |
| `llm:qwen3.5:4b` | 197 | 47 s | 267 ms | section 14 |
| greedy (scripted) | 201 | 63 s | 0 ms | section 11 |
| `llm:gemma4:e4b` | 230 | 55 s | 253 ms | section 14 |
| `llm:phi4-mini` | 232 | 55 s | 187 ms | section 14 |
| `openai:gpt-6-luna` (OpenAI Decisions API) | 237 | 73 s | 168 ms | section 21 |
| `pacman-0.8b-qwen` | 425 | 86 s | 55 ms | section 18 |

The `pacman-0.8b-qwen` row is the mean of three 10-game runs: 441, 467, and 366 pellets. The `openai:gpt-6-luna` row is the mean of two 10-game runs, 239 and 235 pellets, each run at the same time as a Jev run on the same seeds; Jev scored 188 and 178 in those runs, so its row stands. Paired by seed and run, GPT-6 Luna ate 54 more pellets per game than Jev, with a standard error of 15 (section 21).

For commands to run ladder evaluations, see [guides/evaluate.md](guides/evaluate.md).

## The distilled model

`pacman-0.8b-qwen` is a LoRA fine-tune of Qwen3.5-0.8B, the base of `tev1:0.8b`, served as Q8_0 GGUF through Ollama's `/v1/systemone`. Its teacher, Qwen3.8-27B, chose a move for each training state (31,570 rows after deduplication) while seeing one simulated 5-second future per option (`teacher-peek5s`). That lookahead was used only to make labels. The student reads `features` only, like every player on the ladder (section 18).

| Measure | `pacman-0.8b-qwen` | Compare with | Source |
|---|---|---|---|
| Realtime pellets (3-run mean) | 425 | `tev1:0.8b` 69, Jev 178 | section 18 |
| Lockstep pellets | 417 | the teacher without lookahead, thinking off: 301 (seeds 100-102) | sections 13, 18 |
| Decision p50, M3 Ultra | 55 ms | `tev1:0.8b` 62 ms | sections 11, 18 |
| Decision latency, iPhone 16 Pro Max | about 505 ms | | [guides/iphone.md](guides/iphone.md) |
| Agreement with the teacher | 61.9% | `tev1:0.8b` 34.3% | section 18 |
| JevBench accuracy (194 items) | 0.53 | `tev1:0.8b` 0.66 | section 18 |

Sections 22-24 replicated and varied this recipe on NVIDIA H100s. A full rebuild with Qwen3.8-27B in BF16 and vLLM serving (`scripts/systemone_vllm.ts` with `--logprobs-mode processed_logprobs`) matched the published model on 50 paired lockstep games over seeds 100-149 (-3.9 pellets per game, standard error 29.9). Relabeling the same states with DeepSeek-V4-Flash (91.1% agreement with Qwen) trained a student that matched pellets and survival in both clocks while eating about 3 fewer frightened ghosts per game (section 23). Training on Qwen's token probabilities (`TEACHER_LOGPROBS=1`, `build_sft.py --target probs`) or on the average of both teachers' choices stayed within 1.1 to 1.3 standard errors of the published model over 50 to 100 evaluation seeds (section 24).

The weights are on Hugging Face at [grapeot/decision-pacman-0.8b-GGUF](https://huggingface.co/grapeot/decision-pacman-0.8b-GGUF), with a Modelfile that carries the system prompt the model was trained under. [guides/run-models.md](guides/run-models.md) shows how to import them into Ollama. The training data is not published; to rebuild the model, follow [guides/distill.md](guides/distill.md).

## Lockstep results

Under the lockstep clock, the simulation pauses until the model returns an answer. These results isolate move quality from latency and do not belong on the ladder.

| Player | Seeds | Mean pellets | Decision p50 | Source |
|---|---|---|---|---|
| Qwen3.8-27B, features, thinking off | 100-102 | 301 | - | sections 10, 13 |
| Qwen3.8-27B, features, thinking on | 100-102 | 531 | 2.4 s (p90 8.6 s) | section 13 |
| `pacman-0.8b-qwen` | 100-109 | 417 | - | section 18 |

The Qwen evaluations covered 3 games on seeds 100-102 (individual scores: 361, 361, 180 with thinking off). Qwen ran on an RTX 5090 (NVFP4) over the local network.

## Agreement with the teacher

This offline test measures how often a model selects the teacher's move on 1,000 held-out validation states (seeds 900-999, never used for training). It is a secondary check; play in the game carries more weight.

| Player | Agreement | Source |
|---|---|---|
| `tev1:0.8b` | 34.3% | section 18 |
| `tev1:4b` | 38.2% | section 18 |
| `pacman-0.8b-qwen` | 61.9% | section 18 |

The teacher-labeled set was scored for these models only. The other players were scored against a second reference: the best option of the rollout search `oracle-5s` (see the FAQ below) on 1,000 validation states drawn the same way.

| Player | Agreement with the search | Source |
|---|---|---|
| `tev1:0.8b` | 31.8% | section 11 |
| `tev1:4b` | 38.4% | section 11 |
| Jev 1.13.0 | 39.9% | section 11 |
| `llm:gemma4:e4b` | 43.6% | section 14 |
| `llm:phi4-mini` | 44.0% | section 14 |
| `llm:qwen3.5:4b` | 45.4% | section 14 |
| `pacman-0.8b-qwen` | 55.3% | section 18 |

## Generality outside Pac-Man

Accuracy on 194 public JevBench decision items (non-Pac-Man tasks including support routing, intent detection, policy checks, and severity classification; items exceeding 4,000 characters skipped).

| Model | Accuracy | Source |
|---|---|---|
| `tev1:4b` | 0.81 | section 15 |
| `tev1:0.8b` | 0.66 | section 15 |
| `pacman-0.8b-qwen` | 0.53 | section 18 |
| Chance baseline | 0.32 | section 15 |
| Jev 1.13.0 (JevBench's published run) | 0.89 | section 15 |

Jev's 0.89 is computed from JevBench's published per-item results on the same 194 items, not measured here. The evaluation script is `scripts/eval_general_decisions.py`.

## Where decision time goes

Latency medians measured on 20 game prompts of approximately 383 tokens (section 17). The 0.8B rows were measured on `pacman-0.8b`, an earlier fine-tune with the same base, size, and Q8_0 quantization as `pacman-0.8b-qwen`.

| Metric | M3 Ultra, Ollama | RTX 5090, llama-server | RTX 5090, llama-bench |
|---|---|---|---|
| `tev1:4b` prefill | 178 ms | 50 ms | 25 ms |
| `tev1:4b` each further output token | 12.0 ms | 4.8 ms | 4.5 ms |
| `tev1:4b` end to end, 1 output token | 197 ms | 124 ms | - |
| fine-tuned 0.8B prefill | 41 ms | 17 ms | 11 ms |
| fine-tuned 0.8B each further output token | 5.5 ms | 2.1 ms | 2.0 ms |
| fine-tuned 0.8B end to end, 1 output token | 54 ms | 47 ms | - |

A decision is prefill-bound. A decision model reads its answer directly from option logits in one forward pass, whereas a chat model generating a ~6-token JSON response spends an additional ~72 ms on the M3 Ultra and ~27 ms on the RTX 5090 in decoding steps.

## Variance and caveats

Realtime scores for the same model varied by up to about 150 pellets across 10-game runs (section 18). `pacman-0.8b-qwen` recorded runs of 441, 467, and 366 pellets. The 2026-10-01 runs shared the host Mac with heavy unrelated background workload (load average 15-36); the models compared that day were alternated to face matching load conditions. Compare realtime scores within the same session rather than across days.

Single games show wide variance. Jev's 10 evaluation games ranged from 103 to 231 pellets.

Lockstep games also vary across hardware and batching when near-tie options flip. `pacman-0.8b-qwen` scored 417 pellets on seeds 100-109 on the M3 Ultra and 401 on an H100, and parallel and sequential vLLM runs of the rebuilt student diverged on 7 of 10 seeds (section 22). Compare close models over 50 to 100 paired seeds on one stack.

Hosted players depend on the network and the provider's load. OpenAI's Decisions API is a public beta: one GPT-6 Luna game ran at a p50 of 301 ms against about 160 ms for the rest, and Jev answered one request with HTTP 529 (overloaded). Both were run at the same time so that they shared these conditions.

The realtime clock charges latency as game time, so it handicaps slower players, such as the chat models at 187-267 ms per decision.

Repeated identical requests hit inference caching and return 5-10x faster, so reported latency numbers require fresh game states.

The clips below are single 30-second games. They illustrate play; they do not measure it.

## Clips

The browser clips record 30 seconds of gameplay on seed 100 at 1x speed. The iPhone clip is a 29-second screen recording at 1x on a random seed.

| File | Player | Score | Pellets | Lives lost |
|---|---|---|---|---|
| [`media/demo_tev1_08b.mp4`](media/demo_tev1_08b.mp4) | `tev1:0.8b` | 710 | 71 | 2 |
| [`media/demo_tev1_4b.mp4`](media/demo_tev1_4b.mp4) | `tev1:4b` | 1,250 | 101 | 1 |
| [`media/demo_jev.mp4`](media/demo_jev.mp4) | Jev 1.13.0 (hosted) | 900 | 86 | 1 |
| [`media/demo_qwen35_4b.mp4`](media/demo_qwen35_4b.mp4) | `llm:qwen3.5:4b` | 1,370 | 109 | 0 |
| [`media/demo_phi4_mini.mp4`](media/demo_phi4_mini.mp4) | `llm:phi4-mini` | 1,710 | 143 | 2 |
| [`media/demo_pacman_08b_qwen.mp4`](media/demo_pacman_08b_qwen.mp4) | `pacman-0.8b-qwen` | 3,320 | 176 | 0 |
| [`media/iphone_pacman_08b_qwen.mp4`](media/iphone_pacman_08b_qwen.mp4) | `pacman-0.8b-qwen` on an iPhone 16 Pro Max | 2,500 | 174 | 0 |

## Records with lookahead in the input

In these runs the player's input included engine rollouts (lookahead). They are kept as records and as evidence of label quality. They are not compared with the ladder.

| Player | Clock | Mean pellets | Source |
|---|---|---|---|
| Qwen3.8-27B (`teacher-peek5s`), the teacher as it labeled | lockstep | 1,396 | section 13 |
| Student B (`pacman-0.8b-qwen-peek`) | lockstep | 1,387 | section 18 |
| Student B (`pacman-0.8b-qwen-peek`) | realtime | 765 | section 18 |
| Jev (`features-peek5s`) | realtime | 567 | section 16 |
| `tev1:4b` (`features-peek5s`) | realtime | 212 | section 16 |

Student B was trained on the same teacher labels as `pacman-0.8b-qwen` but reads the lookahead encoding, so it is a record of how much of the teacher a 0.8B can copy when it sees what the teacher saw.

## FAQ: the search oracle and the first student

The game has an exact simulator, so a search can play it well. `oracle-5s` (`src/agent/oracle.ts`) copies the game for each option and rolls the engine forward 5 seconds with greedy play at later junctions. It sees the future at decision time, so it is a record, not a ladder player. Real tasks rarely have a simulator, which is why the project's teacher is a general LLM. The same rollouts are what the teacher saw when labeling.

| Measure | Value | Source |
|---|---|---|
| `oracle-5s`, realtime, seeds 100-109 | 1,109 pellets at about 5 ms of CPU per decision | section 10 |
| `oracle-3s`, `oracle-8s` | 1,009 and 1,011 pellets | section 10 |
| Teacher's agreement with the search's best option | 90.7% of training states, 92.0% of validation states, 88.3% on section 11's 1,000 states | section 18 |
| `pacman-0.8b`, an earlier 0.8B trained on the search's labels (31,585 states) | 386 pellets, mean of 3 runs (456, 300, 400), 59-63 ms | sections 11, 18 |
| `pacman-0.8b` agreement with the search | 53.9% | section 11 |
| `pacman-0.8b` JevBench accuracy | 0.49 | section 18 |

The first student and the LLM-distilled one play at the same level: the gap between 386 and 425 is smaller than the run-to-run spread. Section 18 also trained a third student on the search's best option as hard labels (406 pellets, one run).
