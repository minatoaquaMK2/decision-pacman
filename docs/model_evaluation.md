# Model evaluation

This document records how we chose the in-game model, the fine-tuning target, and the teacher, and what the evidence does and does not support. It will grow as later stages produce in-game results.

> This file is the lab notebook, kept in the order the experiments were run, and section numbers are stable. Current headline numbers and their caveats are in [`results.md`](results.md); how to reproduce them is in [`guides/evaluate.md`](guides/evaluate.md) and [`guides/distill.md`](guides/distill.md).

## Summary

- **In-game default: `tev1:4b`.** It was the best off-the-shelf trade-off on this machine: about 160 ms per decision and 0.88 accuracy on the probe, faster and more accurate than `nimble`.
- **Fine-tuning target: a 0.8B model.** `tev1:0.8b` answers in about 65 ms, which is close to real time, but scores 0.40 on the probe, which is near chance. Fast and weak is the case where fine-tuning has the most to show.
- **Labels: a search oracle or a 27B LLM with lookahead, with the same result.** The first student learned from a search oracle that rolls the engine forward 5 s per option (1,109 pellets at 5 ms of CPU). Qwen3.8-27B without lookahead lost all its lives within 39 to 93 s, so it was not a usable teacher on the encoded state alone; given lookahead at labeling time it agreed with the oracle on 88-92% of states, and a features-only student trained on its labels plays at the oracle-trained student's level (section 18).
- **Fine-tuning works.** A 0.8B model fine-tuned on 31,585 oracle-labeled states (`pacman-0.8b`) averaged 456 pellets and 94 s per game at 63 ms per decision. That is 6.6 times off-the-shelf `tev1:0.8b` at the same speed, 3 times `tev1:4b`, and more than twice the scripted greedy rule. The oracle it learned from still averages 1,109.
- **The specialization is not free.** On 194 public JevBench decisions that are not Pac-Man (support routing, intent, policy checks, severity), `pacman-0.8b` scored 0.49 against 0.66 for `tev1:0.8b` and 0.81 for `tev1:4b`, with chance at 0.32. It loses most on the easy and standard tiers, where `tev1:0.8b` is competent, and its probabilities there stay close to uniform. On the hard tier both 0.8B models are near chance.
- **Hosted Jev sits between `tev1:4b` and the greedy rule.** Through TypeSafe's API, Jev 1.13.0 averaged 178 pellets and 52 s per game at 115 ms per decision, and agreed with the oracle on 39.9% of held-out states. The fine-tuned 0.8B model beats it on both.
- **An ordinary small chat model plays as well as Jev.** Given the same facts, a JSON schema restricted to the legal options, and thinking off, stock `phi4-mini` averaged 232 pellets at 187 ms per decision, `gemma4:e4b` 230, and `qwen3.5:4b` 197, against Jev's 178, all in real time with no failed answers. All three also agreed with the oracle more often than Jev (44-45% against 39.9%).
- **OpenAI's Decisions API also beats Jev on the same question.** GPT-6 Luna, asked through `/v1/decisions` with the facts and option list Jev gets, averaged 239 and 235 pellets in two runs at about 160 ms per decision, 54 pellets per game above Jev run at the same time (standard error 15, 20 paired games), with no failed answers, at about $0.012 per game (section 21).
- **Lookahead labels data; it never enters a player's input.** Showing a model one simulated 5-second future per option lets it see a near-exact future, because the engine is deterministic apart from frightened ghosts. Results with lookahead in the input (sections 13, 16, and 18's student B) are recorded but are not comparable with players that see only the current state, and are not used as player comparisons. Lookahead is used to make labels: Qwen3.8-27B choosing with it agreed with the search oracle's best option on 88-92% of states, and played 1,396 pellets in lockstep.
- **The 27B teacher distills into a fast 0.8B that sees only the current state.** Trained on Qwen3.8-27B's lookahead labels with the features input, `pacman-0.8b-qwen` averaged 425 pellets over three realtime runs at 55 ms per decision, more than twice Jev's 178, and 0.53 on the JevBench items. The oracle-trained `pacman-0.8b`, run the same day, scored 300 and 400 (456 earlier): a general LLM can replace the hand-built search as the teacher.
- **State representation matters more than model size.** Compact per-direction facts beat a full ASCII board in both latency and decision quality. Quantization and prefix caching did not reduce latency.

The probe uses synthetic scenarios with one clearly correct move each. It measures latency, output reliability, and basic judgment. It does not measure how well a model plays. Experiments 8 to 10 measure play in the game itself and carry more weight.

## Setup

- **Hardware:** Apple M3 Ultra for all local decision models. The teacher ran on a separate machine with an RTX 5090 (NVFP4 weights), reached over the local network.
- **Runtimes:** Ollama 0.35.0 (`/v1/systemone`) and ollaya 0.7.5 (same wire format, port 11435). The teacher was served through an OpenAI-compatible chat API.
- **Keep-alive:** Models stayed loaded (`keep_alive: -1`). Each run started with one warm-up request, which is excluded from the numbers.
- **Fresh states:** Every timed request used a newly randomized state. An exact repeat of a request returns in 16–60 ms because of caching, so a benchmark that repeats states overstates speed by 5–10x.

## Experiments

### 1. API smoke test and CORS

The support-ticket example from Ollama's documentation returned the documented answer (`bug`, p = 0.978). A request with `Origin: http://localhost:5173` returned `Access-Control-Allow-Origin` for that origin, so a page served from localhost can call Ollama directly.

### 2. Full ASCII board

We sent the full 28×31 board with Pac-Man and ghosts drawn in, using only legal moves as options, on 12 random positions.

| Model | Input tokens | Latency (cold) | Confidence |
|---|---|---|---|
| nimble | ~500 | ~430 ms | 0.00–0.09 |
| tev1:4b | ~460 | ~240 ms | 0.02–0.04 (3 positions logged) |
| tev1:0.8b | ~460 | ~80 ms | 0.01–0.03 (3 positions logged) |

Confidence near zero means the probability mass is spread almost evenly over the options. The models could not read the board as a picture.

### 3. Prefix caching

We tested the same ~530-token board with the static maze first and the changing part last, against the reverse order. Latency went from ~455 ms to ~420 ms, about an 8% saving. Latency scales with input tokens at about 0.8 ms per token for `nimble`, so the only effective lever is a smaller state.

### 4. Synthetic decision probe (`scripts/probe_decision_models.py`)

Each of 40 scenarios gives, for every legal exit, the distance to the nearest ghost and the pellets within 5 steps, plus whether ghosts are frightened. One exit is clearly best. In normal mode, the other exits have a ghost 1–3 steps away. In frightened mode, the best exit leads toward the edible ghost and the others lead to nothing. About a third of the scenarios are frightened. The instructions state the rules, including that frightened ghosts are edible. The state is compact JSON of about 130–250 tokens.

| Runtime | Model | Latency p50 | Latency p90 | Accuracy | Frightened cases |
|---|---|---|---|---|---|
| Ollama | tev1:4b (q8) | 156 ms | 188 ms | **0.88** | 13/13 |
| Ollama | tev1:4b (q4_K_M) | 161 ms | 188 ms | 0.88 | 13/13 |
| Ollama | nimble 9B (q8) | 270 ms | 299 ms | 0.68 | 12/13 |
| Ollama | nimble 9B (q4_K_M) | 286 ms | 306 ms | 0.60 | 12/13 |
| Ollama | tev1:0.8b (q8) | 67 ms | 83 ms | 0.42 | 9/13 |
| Ollama | tev1:0.8b (bf16) | 65 ms | 78 ms | 0.40 | 8/13 |
| ollaya | laya:en (421M encoder) | 20 ms | 25 ms | 0.40 | 11/13 |
| ollaya | kev:0.8b | 729 ms | 970 ms | 0.40 | 9/13 |
| ollaya | winnow:e4b (GGUF, Metal) | 154 ms | 170 ms | 0.78 | 9/13 |
| ollaya | decider:0.8b | 330 ms | 447 ms | 0.35 | 5/13 |
| ollaya | decider (2B) | 681 ms | 955 ms | 0.60 | 8/13 |
| ollaya | decider:4b | 2,186 ms | 2,724 ms | 0.65 | 12/13 |
| ollaya | kev (4B) | 2,052 ms | 2,672 ms | 0.50 | 13/13 |
| ollaya | kev:9b | 3,940 ms | 4,721 ms | 0.57 | 13/13 |

Random choice over 2–4 legal options scores about 0.36 on this set. Observations:

- `tev1:4b` is both faster and more accurate than `nimble`. Its advantage is in normal-mode scenarios, where it has to trade pellets against a nearby ghost.
- q4 quantization did not speed anything up. A decision outputs a single token, so the time goes into reading the input, which is limited by compute, not memory bandwidth.
- The 0.8B-class models and `laya:en` are near chance. `laya:en` does well when ghosts are frightened (move toward the ghost) and poorly otherwise.
- On this Mac, ollaya runs `kev` and `decider` on the CPU, which puts them at 0.3–4 s per decision, too slow for real-time play here whatever their accuracy. `winnow:e4b` ships as GGUF and runs on llama.cpp with Metal. It is the one ollaya model in `tev1:4b`'s latency range, at 0.78 against 0.88.

### 5. Plain English instead of JSON

We rendered the same facts as sentences ("Going left: a ghost 2 steps away, 3 pellets in the next 5 steps."). It did not help. `laya:en` went from 0.40 to 0.47, and `tev1:4b` dropped from 0.88 to 0.78. We keep compact JSON.

### 6. Teacher: Qwen3.8-27B (`scripts/probe_teacher.py`)

Same 40 scenarios, same rules, same JSON state. The prompt asks for `{"move": ..., "reason": ...}`. Temperature 0.

| Mode | Latency p50 | Latency p90 | Parseable | Legal | Accuracy | Output tokens |
|---|---|---|---|---|---|---|
| Thinking off | 270 ms | 309 ms | 40/40 | 40/40 | 0.97 | ~36 |
| Thinking on (concurrency 4) | 1,135 ms | 1,553 ms | 40/40 | 40/40 | 1.00 | ~190 |
| Thinking off, concurrency 8 (80 requests) | 405 ms | 731 ms | 80/80 | 80/80 | 0.97 | ~36 |

Throughput at concurrency 8 was 16 labels per second. The server does not support `response_format: json_schema`, because it has no constrained decoding, but a prompt that asks for JSON was parseable in all 120 calls. The two misses with thinking off both went for more pellets at the cost of ghost distance, which is the same kind of trade-off where the small models fail.

### 7. Prior art: Ollama's Pac-Man replay

Ollama's launch post embeds a recorded game as JSON: `nimble:9b-int4`, a 19×21 maze, three ghosts, one decision per tile step (turn-based), 187 moves, 91 ms mean latency on an M5 Max, median confidence 0.58, ending with Pac-Man caught. It confirms that the legal-moves-only option set works. It ships no engine, prompt, or frontend to reuse.

### 8. In-game results (Stage 1, headless)

`scripts/run_headless.ts`, 1x speed, realtime clock (model latency turns into game time), features encoder, junction lookahead, seeds starting at 100. Level 1 has 244 pellets.

| Policy | Games | Mean pellets | Mean score | Mean survival | Decision p50 | Stale answers |
|---|---|---|---|---|---|---|
| random | 10 | 30.3 | 311 | 37 s | 0 ms | 0% |
| greedy (scripted) | 10 | 200.7 | 2,623 | 63 s | 0 ms | 0% |
| tev1:4b | 3 | 168.7 | 1,833 | 57 s | 207 ms | 10% |

tev1:4b plays far above random and below the scripted baseline, which reads the same facts with hand-written rules. This is the gap the teacher data and fine-tuning stages are meant to close for a 0.8B model. Three games are enough to show that play works, not to rank models. Stage 4 will use more seeds.

### 9. Qwen3.8-27B as a player (`gen_teacher_data.ts`, lockstep)

The teacher gets the same encoded state as the student and answers `{"move", "reason"}`. Lockstep means the game waits for each answer, so latency does not count against it.

30 seconds per game, seeds 1001-1003:

| Qwen sees | Thinking | Pellets (3 games) | Mean pellets | Deaths (total) | Unparseable |
|---|---|---|---|---|---|
| Features | off | 155, 182, 170 | 169 | 2 | 0 |
| Features | on | 165, 181, 182 | 176 | 0 | 13 of ~500 |
| Full ASCII map (`ascii-full`) | off | 105, 107, 85 | 99 | 6 | 0 |

The full-map run with thinking on was stopped after 48 minutes, too slow to matter.

**Qwen misreads the ASCII map.** Its reasons mention dead ends, which this maze does not have, and a tunnel on row 29, when the tunnel is on row 14. It also headed for a distant power pellet when one sat on the same row. Reading vertical relationships from a grid serialized as text is unreliable: tiles above and below each other sit dozens of tokens apart.

**30 seconds hides the difference between players.** In 5-minute lockstep games on seeds 100-102, Qwen (features, thinking off) ate 361, 361, and 180 pellets. It lost all three lives within 74, 93, and 39 seconds.

### 10. Search baselines

**Survey.** A literature survey of simple, non-RL Pac-Man agents found a consistent lesson. The strongest simple agents rely on two things: an accurate model of ghost behavior, and a "safe path" test (Pac-Man reaches the next junction before any dangerous ghost). Supporting evidence:

- A model-based Ms. Pac-Man agent predicted ghost moves with 94.6% accuracy and averaged 38,172 points on the real game, above the competition record of the time. Sources: [Cornell news](https://news.cornell.edu/stories/2017/01/engineers-eat-away-ms-pac-man-score-artificial-player), survey in IEEE ToG 2018 ([ResearchGate](https://www.researchgate.net/publication/321821472)).
- The MCTS agent that won CIG'12 averaged 82,689 against the framework's baseline ghosts. Its ablations show that most of its strength comes from rule-based, safety-aware playouts: random playouts cut the score by 60-83%. Source: [Pepels, Winands and Lanctot](https://dke.maastrichtuniversity.nl/m.winands/documents/CIG2012_paper_106.pdf).
- Rule-based agents with A* over ghost-weighted path costs (ICE Pambush) won CEC'09 with a 13,059 average. Source: [Matsumoto et al.](https://www.ice.ci.ritsumei.ac.jp/~ruck/PAP/jsst09-matsumoto.pdf).
- The UC Berkeley CS188 Pac-Man agents are teaching templates, not strength benchmarks. The project only asks for more than half wins against one random ghost on a small layout. Source: [CS188 project 2](https://inst.eecs.berkeley.edu/~cs188/fa24/projects/proj2/).
- Our `greedy` rule matches the framework's starter agent: flee, chase edible ghosts, else go to the nearest pellet.

Published scores come from real Ms. Pac-Man or the Java competition framework, so they are not comparable with ours. Only the ranking of techniques transfers. Our engine is easier than both: ghost moves are deterministic except in frightened mode, so an agent can simulate the future exactly.

**Our oracle** (`src/agent/oracle.ts`). For each option, copy the game, play the option, and run the real engine forward for a fixed horizon, with greedy play at later junctions. The value is the score gained, minus a large penalty for a death that shrinks the later the death happens, minus the distance to the nearest pellet at the end.

Evaluation seeds 100-109, realtime clock, 1x speed, 5-minute cap:

| Horizon | Mean pellets | Mean score | Games reaching the cap | Decision p50 |
|---|---|---|---|---|
| 3 s | 1,009 | 16,546 | 10 of 10 | 3 ms |
| **5 s** | **1,109** | **19,789** | 10 of 10 | 5 ms |
| 8 s | 1,011 | 15,858 | 10 of 10 | 7 ms |

A level has 244 pellets, so 1,109 is about four and a half levels. Horizons past 5 seconds do not help, probably because the greedy playout policy gets less reliable further out. That explanation is not tested.

5-minute lockstep head-to-head, seeds 100-102:

| Player | Pellets | Result |
|---|---|---|
| oracle-3s | 971, 1,178, 1,211 | 1 death per game, all reached the cap |
| Qwen3.8-27B (features, thinking off) | 361, 361, 180 | all lives lost at 74, 93, 39 s |

### 11. Fine-tuned 0.8B (`pacman-0.8b`)

`Qwen/Qwen3.5-0.8B` fine-tuned with LoRA on oracle-labeled states in the exact prompt format Ollama uses (see RFC, Stage 3). It is served as Q8_0 GGUF through Ollama like the stock models.

**Agreement with the oracle**, 1,000 validation states from seeds 900-999 (never used for training), scored through Ollama. "Decisive" means the oracle's soft target puts more than 0.9 on one option.

| Model | Agreement | On decisive states | Cross-entropy | Latency p50 |
|---|---|---|---|---|
| Qwen3.5-0.8B before training (training-side eval, 2,000 states) | 28.8% | 30.6% | 1.43 | - |
| `tev1:0.8b` | 31.8% | 29.5% | 1.22 | 73 ms |
| `tev1:4b` | 38.4% | 40.7% | 1.29 | 205 ms |
| Jev 1.13.0 (hosted) | 39.9% | 40.9% | 1.41 | 237 ms |
| **`pacman-0.8b`** | **53.9%** | **57.5%** | **1.05** | **59 ms** |

Validation agreement rose from 29% to 53% within 1,200 of 1,975 steps and then flattened. The plateau is likely the information gap: many oracle decisions depend on facts the features encoder does not include.

**In-game**, evaluation seeds 100-109, realtime clock, 1x speed, 5-minute cap:

| Player | Mean pellets | Mean score | Mean survival | Levels cleared | Decision p50 | Stale |
|---|---|---|---|---|---|---|
| random | 30 | 311 | 37 s | 0 of 10 | 0 ms | 0% |
| `tev1:0.8b` | 69 | 716 | 35 s | 0 of 10 | 62 ms | 4% |
| `tev1:4b` | 148 | 1,660 | 55 s | 0 of 10 | 202 ms | 11% |
| Jev 1.13.0 (hosted) | 178 | 2,482 | 52 s | 0 of 10 | 115 ms | 3% |
| greedy (scripted) | 201 | 2,623 | 63 s | 0 of 10 | 0 ms | 0% |
| **`pacman-0.8b`** | **456** | **7,467** | **94 s** | **8 of 10** | **63 ms** | **3%** |
| `oracle-5s` (teacher) | 1,109 | 19,789 | 300 s (cap) | 10 of 10 | 5 ms | 0% |

Per game, `pacman-0.8b` ate 239, 400, 417, 649, 482, 683, 587, 420, 240, and 446 pellets. The best game reached level 3.

**Parity between training and serving.** On 50 validation states, the Q8_0 model in Ollama and the bf16 weights in PyTorch agree to within 0.027 in probability (median 0.010), and pick the same option on 48 of 50. The two that differ are near-ties.

**Recordings**, all seed 100 at 1x for 30 s:

| Clip | Model | Score | Pellets | Lives lost |
|---|---|---|---|---|
| `docs/media/demo_tev1_08b.mp4` | `tev1:0.8b` | 710 | 71 | 2 |
| `docs/media/demo_tev1_4b.mp4` | `tev1:4b` | 1,250 | 101 | 1 |
| `docs/media/demo_jev.mp4` | Jev 1.13.0 (hosted) | 900 | 86 | 1 |
| `docs/media/demo_qwen35_4b.mp4` | `llm:qwen3.5:4b` (plain chat model, JSON) | 1,370 | 109 | 0 |
| `docs/media/demo_pacman_08b.mp4` | `pacman-0.8b` | 2,660 | 170 | 0 |

In the `tev1:0.8b` clip, the option probabilities stay close to uniform (confidence 0.00-0.04).

### 12. Jev, TypeSafe's hosted decision model

Jev is served at `https://api.typesafe.ai/v1/systemone` with the same request and response format as Ollama's endpoint, so the game calls it unchanged. Only the base URL, the model name (`jev-latest`, which answered as `jev-1.13.0`), and a bearer key differ. The API rejects Ollama's `keep_alive` field, so the client leaves it out for hosted endpoints. Run it with `TYPESAFE_API_KEY` set: `npm run headless -- --policy jev`, or `VITE_DECISION_BASE_URL=https://api.typesafe.ai npm run dev` and `?model=jev-latest` in the browser. The dev server adds the key to proxied requests, so it never reaches the page.

**Latency.** Over the internet from the test machine, p50 was 106-123 ms per game in the headless run. The agreement run, which sent requests while the games were also running, measured p50 237 ms and p90 298 ms. That is comparable to `tev1:4b` running locally, which pays for its latency with local compute instead of network.

**Play.** Per game, Jev ate 208, 173, 165, 198, 125, 203, 103, 231, 193, and 179 pellets, and lost all three lives in every game. It plays better than `tev1:4b` (148) and slightly worse than the greedy rule (201).

**Tokens.** The same state costs about 605 input tokens on Jev against about 380 on the local models. A trivial request costs 325, so the hosted service adds roughly 300 tokens of its own per call. The 10 games used 2.05 million input tokens over 3,368 decisions.

### 13. Qwen3.8-27B with thinking, and with lookahead

Two ways to give the general model more to work with, both in lockstep (the game waits for each answer), 1x speed, 5-minute cap, features encoder.

- **Thinking** (`teacher-think`): the same prompt as section 9, with thinking on.
- **Lookahead** (`teacher-peek5s`, thinking off): for each option, the engine plays that option and then greedy moves for 5 s, exactly as the oracle's rollout does. The prompt adds the outcome per option as facts: seconds until death (or none), pellets, power pellets, ghosts eaten, and points. The model makes the choice.

| Player | Seeds | Mean pellets | Mean score | Mean survival | Decision p50 | Failed answers |
|---|---|---|---|---|---|---|
| Qwen, thinking off (section 10) | 100-102 | 301 | - | 69 s | - | - |
| Qwen, thinking on | 100-102 | 531 | 7,217 | 114 s | 2.4 s (p90 8.6 s) | 34 of 2,167 |
| `oracle-5s` (realtime) | 100-109 | 1,109 | 19,789 | 300 s (cap) | 5 ms | - |
| **Qwen, lookahead** | 100-109 | **1,396** | **22,054** | 299 s | 309 ms | 11 of 15,668 |

Per game, Qwen with lookahead ate 1,457, 1,372, 1,364, 1,321, 1,427, 1,412, 1,363, 1,391, 1,420, and 1,431 pellets, and every game reached level 6. Nine games reached the cap; seed 102 lost its last life at 292 s. Failed answers fall back to the previous intent.

Both players see the same rollouts. The oracle reduces each to one number (points, minus a large penalty for dying, minus the distance to the nearest pellet) and takes the highest. The model reads the outcomes and chooses. That it does better says the hand-written value function is the oracle's weak part. Which trade-offs the model makes differently has not been analyzed.

Thinking alone helps (301 to 531 pellets) but costs about 9 times the latency, and the model still misjudges danger from the encoded facts alone.

### 14. Plain small chat models with structured output

Does Jev, or Tev1, do something an ordinary small model cannot? `llm:<model>` (`src/agent/llm.ts`) asks a general chat model through Ollama's `/api/chat` with the facts Jev sees: the features state, the same instructions, and the same options, in the teacher's prompt from section 9 without the reason field. A JSON schema restricts `move` to the legal options, so every answer is legal. Thinking is off, temperature is 0, and the model stays loaded. Option probabilities come from the token logprobs at the position where the move is written, renormalized over the options. The clock is realtime, as it was for Jev.

The models are stock Ollama library builds (Q4_K_M), pulled for this test:

- `qwen3.5:4b` (4.7B): the base model that Tev1's 4B LoRA is trained on.
- `gemma4:e4b` (7.5B stored, about 4B used per token).
- `phi4-mini` (3.8B).

**In-game**, evaluation seeds 100-109, realtime clock, 1x speed, 5-minute cap, features encoder:

| Player | Mean pellets | Mean score | Mean survival | Levels cleared | Decision p50 | Stale | Failed answers |
|---|---|---|---|---|---|---|---|
| random | 30 | 311 | 37 s | 0 of 10 | 0 ms | 0% | - |
| `tev1:4b` | 148 | 1,660 | 55 s | 0 of 10 | 202 ms | 11% | - |
| Jev 1.13.0 (hosted) | 178 | 2,482 | 52 s | 0 of 10 | 115 ms | 3% | - |
| `llm:qwen3.5:4b` | 197 | 2,553 | 47 s | 0 of 10 | 267 ms | 9% | 0 of 1,553 |
| greedy (scripted) | 201 | 2,623 | 63 s | 0 of 10 | 0 ms | 0% | - |
| `llm:gemma4:e4b` | 230 | 3,276 | 55 s | 1 of 10 | 253 ms | 4% | 0 of 1,929 |
| `llm:phi4-mini` | 232 | 3,142 | 55 s | 2 of 10 | 187 ms | 4% | 0 of 2,468 |
| `pacman-0.8b` (fine-tuned) | 456 | 7,467 | 94 s | 8 of 10 | 63 ms | 3% | - |

Per game, `qwen3.5:4b` ate 213, 242, 185, 152, 239, 219, 122, 215, 217, and 169 pellets; `gemma4:e4b` 220, 242, 304, 196, 211, 238, 232, 232, 195, and 230; `phi4-mini` 170, 235, 240, 226, 215, 269, 221, 144, 371, and 231. Every game ended with all three lives lost. Paired by seed against Jev, the mean difference is +20 pellets for `qwen3.5:4b` (standard error 14), +52 for `gemma4:e4b` (17), and +54 for `phi4-mini` (24).

**Agreement with the oracle**, the same 1,000 validation states as section 11, asked the same way as in the game (`training/score_ollama.py --chat`):

| Model | Agreement | On decisive states | Cross-entropy | Latency p50 |
|---|---|---|---|---|
| `tev1:4b` | 38.4% | 40.7% | 1.29 | 205 ms |
| Jev 1.13.0 (hosted) | 39.9% | 40.9% | 1.41 | 237 ms |
| `llm:gemma4:e4b` | 43.6% | 43.6% | - | 245 ms |
| `llm:phi4-mini` | 44.0% | 43.6% | 1.72 | 148 ms |
| `llm:qwen3.5:4b` | 45.4% | 46.4% | 1.67 | 237 ms |
| `pacman-0.8b` | 53.9% | 57.5% | 1.05 | 59 ms |

Observations:

- All three plain models play at least as well as Jev in real time, and none produced a failed answer. `gemma4:e4b` and `phi4-mini` also beat the greedy rule. `qwen3.5:4b`'s lead over Jev is within the noise of 10 games; the other two lead by about two to three standard errors. None comes near the fine-tuned 0.8B model.
- Plain `qwen3.5:4b` beats `tev1:4b`, which is the same base with a decision LoRA, both in play (197 against 148 pellets) and in agreement (45.4% against 38.4%). The two differ in more than the LoRA: chat with a generated JSON answer against System One's scoring of option letters, and Q4_K_M against q8 weights. Which of these accounts for the gap was not tested.
- The plain models are slower than Jev, not faster. A chat answer generates about six tokens (`{"move": "left"}`) after reading about 300 input tokens, which takes 190-270 ms locally against Jev's 115 ms over the network. The realtime clock therefore handicaps them. `qwen3.5:4b` is the slowest of the three and has the highest stale rate (9%).
- Probabilities are available from logprobs for `qwen3.5:4b` and `phi4-mini` (median confidence 0.76 and 0.77). Ollama 0.35.0 returns logprobs only for the first generated token of `gemma4:e4b`, so it has none. The chat probabilities are more overconfident than the System One ones: cross-entropy against the oracle's soft target is 1.67-1.72, against 1.29-1.41. They are fine for the probability bars, not as calibrated scores.
- What this shows: on this task, given the same facts, Jev has no advantage that a stock 4B chat model with constrained JSON output lacks. It does not test Jev's other selling points, such as several questions per request or calibration on other tasks. It also says something about the task: every player that reads only the features encoder, scripted, decision model, or chat model, lands between 148 and 232 pellets, while the fine-tuned 0.8B model reaches 456 from the same facts.

### 15. What specialization cost

`pacman-0.8b` plays far better than `tev1:0.8b`. This experiment measures the other side: how much worse it is at decisions that are not Pac-Man.

**Evaluation set.** The public items of [JevBench](https://github.com/fstandhartinger/jevbench) (MIT), a third-party benchmark for decision models in the same `/v1/systemone` format, at commit `bb05a33`. They are labeled short decisions about refund policies, support messages, tool requests, incident severity, and similar text, posed as yes/no (`noul`), pick-one (`choice`), and ordinal (`score`) questions with 2-6 options. JevBench groups them into three tiers: easy (48 items), standard (72), and hard (111, written to contain traps and multi-step reasoning). The models in Ollama were loaded with a 2,050-token context, so items longer than 4,000 characters were skipped: 37 hard items, leaving 194. `scripts/eval_general_decisions.py` downloads the files, checks their hashes, and sends each item once with JevBench's own question. Each item is scored by argmax, as JevBench does, and by the probability the model gave the correct option. Ollama turns all three question types into the same lettered-choice prompt that `pacman-0.8b` was trained on, so the format itself is not new to it.

**Results.** Accuracy, with the probability on the correct option in parentheses. Chance is the mean of 1 / options per item. Latency is p50 on this set (largest prompt 1,381 tokens).

| Model | Easy (48) | Standard (72) | Hard (74) | All (194) | Latency p50 |
|---|---|---|---|---|---|
| chance | 0.28 | 0.31 | 0.36 | 0.32 | - |
| `tev1:4b` | 1.00 (1.00) | 0.94 (0.88) | 0.57 (0.55) | 0.81 (0.79) | 196 ms |
| `tev1:0.8b` | 1.00 (0.96) | 0.69 (0.66) | 0.41 (0.38) | 0.66 (0.63) | 68 ms |
| **`pacman-0.8b`** | 0.79 (0.37) | 0.39 (0.34) | 0.39 (0.37) | 0.49 (0.36) | 69 ms |
| Jev 1.13.0 (JevBench's published run) | 1.00 | 0.99 | 0.73 | 0.89 | - |

Accuracy above chance, as a share of the room above chance ((accuracy - chance) / (1 - chance)), is 0.73 for `tev1:4b`, 0.50 for `tev1:0.8b`, and 0.24 for `pacman-0.8b` over all 194 items. The Jev row is computed from JevBench's published per-item outcomes (v1.2) on the same 194 items, not measured here.

On the 40-scenario Pac-Man probe (section 4), rerun on the same day, the order reverses:

| Model | Probe accuracy | Probability on the correct move | Mean top probability |
|---|---|---|---|
| `tev1:4b` | 0.88 | 0.64 | 0.69 |
| `tev1:0.8b` | 0.42 | 0.40 | 0.54 |
| **`pacman-0.8b`** | **0.93** | 0.52 | 0.52 |

**Where the loss is.**

- Paired with `tev1:0.8b` item by item, `pacman-0.8b` gets 48 items wrong that `tev1:0.8b` gets right and 15 right that it gets wrong. On easy items the count is 10 lost and none gained; on standard items, 26 lost and 4 gained.
- On the standard tier `pacman-0.8b` is near chance in every family: intent 0.25, routing 0.33, ordinal 0.33, policy 0.42, extraction 0.50, against 0.58-1.00 for `tev1:0.8b`. The exception is judging whether an answer is adequate, where it scores 0.50 against 0.33. On easy items it still picks the right tool on 12 of 12 and the right intent on 7 of 12.
- The hard tier does not separate the two 0.8B models. Both are within a few points of chance (0.41 and 0.39), so specialization had little to lose there. Only `tev1:4b` is clearly above chance on it.
- `pacman-0.8b`'s answers lean toward the first two options. Options A or B are correct on 120 of the 194 items, and it picked A or B on 166. On yes/no items it said yes 85% of the time, against 47% in the labels and 66% for `tev1:0.8b`.
- Its probabilities stay close to uniform. When it is right on easy items, it is right by a small margin: 0.79 accuracy with 0.37 on the correct option. `tev1:0.8b` puts 0.96 there. Part of this is how it was trained: the oracle's soft targets spread probability over near-equal moves, so even on the Pac-Man probe its top option averages only 0.52. Out of domain the spread is larger still, and its probabilities are not useful as confidence.

**What the comparison does and does not show.** `pacman-0.8b` was not made by further training `tev1:0.8b`. It is a LoRA on the Qwen3.5-0.8B base, because Tev1's 0.8B weights have only been seen as Ollama GGUF. Tev1 is itself a LoRA on Qwen3.5. The two models are best read as two adapters on the same family of base model: one trained on general decisions, one only on Pac-Man. The gap above is therefore the general decision training `pacman-0.8b` never received, together with anything the Pac-Man training took from the base model. The base model is not served in Ollama here, so the two parts are not separated. Its remaining skill on easy items probably comes from the base model, which is not tested. Other limits: the set is small (12 items per family), it was run once (answers are deterministic), and it excludes the longest hard items.

The practical reading is that `pacman-0.8b` replaces a general decision model rather than extending one. It is the better choice when the model only ever plays this game, and the worse one when the same model must also route tickets or check policies. Keeping both would mean serving two adapters, or training the Pac-Man data together with general decisions, which was not tried.

### 16. Lookahead for a closed model

Section 13's lookahead reached Qwen through its prompt, so nothing about it needs open weights. The `features-peek5s` encoder puts the same facts into the `/v1/systemone` state, so any decision model can receive them, including a hosted one. It is the features encoding plus `lookahead_5s`: for each option, the outcome of `peek()` (play the option, then greedy moves for 5 s) as seconds until death or null, pellets, power pellets, ghosts eaten, and points. The instructions add one paragraph with section 13's wording: what the rollout is, that frightened ghosts turn at random, that it is one possible future, and what each field means. Like every encoder, it reports outcomes and never says which option is better.

Seeds 100-109, realtime clock, 1x speed, 5-minute cap. Both models play the same encoder; only the model differs.

| Player | Encoder | Mean pellets | Mean score | Mean survival | Mean level reached | Decision p50 | Stale answers | Input tokens per decision |
|---|---|---|---|---|---|---|---|---|
| Jev (section 12) | features | 178 | 2,482 | 52 s | 1.0 | 115 ms | 3.0% | 609 |
| **Jev** | features-peek5s | **567** | **9,150** | 175 s | 2.5 | 111 ms | 2.2% | 865 |
| `tev1:4b` (section 11) | features | 148 | 1,660 | 55 s | 1.0 | 202 ms | 10.7% | 368 |
| `tev1:4b` | features-peek5s | 212 | 2,907 | 68 s | 1.0 | 260 ms | 8.6% | 578 |
| `oracle-5s` (section 10) | features | 1,109 | 19,789 | 300 s (cap) | 4.7 | 5 ms | 0% | - |
| Qwen3.8-27B, lockstep (section 13) | features + lookahead in the prompt | 1,396 | 22,054 | 299 s | 6.0 | 309 ms | - | 471 (chat prompt) |

Per game, Jev with lookahead ate 432, 730, 481, 418, 972, 966, 301, 671, 461, and 234 pellets. Every game ended with all three lives lost; two (seeds 104 and 105) lasted to about the 5-minute mark and reached level 4. `tev1:4b` with lookahead ate between 171 and 237 pellets and never cleared level 1.

**Jev gains more than threefold.** The same closed model, with no change but its input, went from 178 to 567 pellets and from 52 s to 175 s. That is above the greedy rule (201) and the fine-tuned `pacman-0.8b` on features (456). Its latency did not change: the 256 extra tokens cost the hosted service nothing measurable.

**It reads the facts, imperfectly.** In 5,760 states where at least one option died in its rollout and at least one survived, Jev chose a dying option 12.8% of the time (`tev1:4b`: 16.8%). These are mostly deaths far down the rollout: Jev chose an option that died within 1.5 s only 20 times. In 441 of its 739 dying choices, that option promised more points than every surviving one. Its probability on those choices was lower (mean confidence 0.43 against 0.65). When every option died, it took the latest death 65% of the time. The rollouts themselves are pessimistic, because they play greedy after the first junction; before most deaths Jev had spent several decisions in states where every rollout died.

**`tev1:4b` gains little.** 148 to 212 pellets is a 43% gain, against more than 200% for Jev and about 360% for Qwen (lockstep). Part of the gap is latency: the extra 210 tokens took its p50 from 202 ms to 260 ms, and an answer applies about eight ticks after the state it was asked about. The rest is how the model reads the same facts.

**Cost.** The 10 Jev games used 9.72 million input tokens over 11,231 decisions, about $0.41 at the $0.042 per million seen on Vercel AI Gateway (section 12, conclusion 7). Games last three times longer, so the run cost about five times the features-only run.

**What this says.** The input knob is open to a closed model, and it took Jev past the fine-tuned student: Jev with lookahead (567) outplays `pacman-0.8b` on plain features (456). How much the knob is worth depends on the model: the same facts gave `tev1:4b` 43%, Jev about 220%, and Qwen about 360%. And for Jev the facts are worth less than a fixed formula over them: `oracle-5s` scores the same rollouts and averages 1,109 pellets. Once the input contains the outcomes, a model earns its place only by weighing them better than a formula does, which Qwen did and Jev did not.

**Caveats.** Qwen's numbers are lockstep, where the game waits for each answer; Jev's and `tev1:4b`'s are realtime. The rollouts run inside the encoder, before the request is timed, so their 5-10 ms of CPU is not charged to game time as the oracle's is. One run per seed; Jev's games vary from 234 to 972 pellets. `tev1:4b` ran on local Ollama with no other games running.

### 17. Where the time goes: prefill, decode, and hardware

`scripts/bench_prefill_decode.py` sends the 20 rendered game prompts in `ios/mac_check/bench_prompts.json` (about 383 tokens each) with 1 output token, as a decision model needs, and with 7, as a chat model writing a short JSON answer needs. Every request starts with a unique tag so no prompt cache is reused. The same Q8_0 GGUF files ran on both machines: Ollama 0.35.0 on the Apple M3 Ultra, and a CUDA build of llama.cpp (`llama-server` with default settings, 8 slots) on one RTX 5090. `llama-bench` (prompt 384, generate 7) gives the 5090's compute-only floor. Medians; prefill and decode are the servers' own timings.

| | M3 Ultra, Ollama | RTX 5090, llama-server | RTX 5090, llama-bench |
|---|---|---|---|
| `tev1:4b` prefill (383 tokens) | 178 ms | 50 ms | 25 ms |
| `tev1:4b` each further output token | 12.0 ms | 4.8 ms | 4.5 ms |
| `tev1:4b` end to end, 1 output token | 197 ms | 124 ms | - |
| `tev1:4b` 8 concurrent requests | 5.4 per s | 7.3 per s | - |
| `pacman-0.8b` prefill | 41 ms | 17 ms | 11 ms |
| `pacman-0.8b` each further output token | 5.5 ms | 2.1 ms | 2.0 ms |
| `pacman-0.8b` end to end, 1 output token | 54 ms | 47 ms | - |
| `pacman-0.8b` 8 concurrent requests | 21 per s | 26 per s | - |

- A decision is prefill-bound. Reading one answer from the logits instead of writing a six-token JSON answer saves six decode steps: about 72 ms on the M3 Ultra and 27 ms on the 5090. This matches section 14, where `qwen3.5:4b` writing JSON took 267 ms against 202 ms for `tev1:4b` reading logits.
- The rest is reading the input, which depends on input length and hardware, not on the output format. The 5090 reads a game state 7 times faster than the M3 Ultra for the 4B model and 3.7 times faster for the 0.8B model.
- `llama-server` with default settings adds about 70 ms (4B) and 30 ms (0.8B) on top of prefill, and it barely batched the 8 concurrent requests. Its end-to-end numbers reflect an untuned server, not the hardware.
- Jev's 115 ms over the internet, with about 605 input tokens, fits a datacenter GPU's prefill of a few tens of milliseconds plus the network round trip.

### 18. Distilling the LLM teacher into 0.8B

Sections 11 and 13 left a gap: `pacman-0.8b` learned from the search oracle, not from the slow general model. This experiment relabels the same states with Qwen3.8-27B plus the 5 s lookahead (`teacher-peek5s`, thinking off) and trains 0.8B students on the labels.

**Labels.** The players behind the original training data are deterministic for a seed, so replaying them on the original seeds (training 1000-1019 oracle, 2000-2029 greedy, 3000-3029 random; validation 900-902, 920-924, 940-944) reproduced the same 57,231 training and 8,047 validation states, now with the full game state for the lookahead. Evaluation seeds 100-109 were not used. The teacher's server does not return logprobs, so labels are hard: 0.9 on the teacher's move and 0.1 spread over the other options. 54,325 requests took about 65 minutes at concurrency 8 (about 14 per second). On these states the teacher picked the oracle's best option 90.7% of the time on training states and 92.0% on validation states; on section 11's 1,000 states, 88.3% (94.9% where the oracle is decisive).

**Students.** All use `pacman-0.8b`'s recipe (Qwen3.5-0.8B, LoRA rank 16, alpha 32, lr 1e-4, batch 32, 2 epochs) on one RTX 5090, borrowed through the GPU lease and restored afterwards.

| Student | Input | Labels | Rows (train / val) | Steps | Time |
|---|---|---|---|---|---|
| A `pacman-0.8b-qwen` | features | teacher, hard | 31,570 / 4,973 | 1,974 | 32 min |
| B `pacman-0.8b-qwen-peek` | features-peek5s | teacher, hard | 38,119 / 6,126 | 2,383 | 63 min |
| C `pacman-0.8b-oracle-hard` | features | oracle's best, hard | same as `pacman-0.8b` | 1,975 | 32 min |

C separates who labels from the label form: it has `pacman-0.8b`'s states and teacher-style hard labels from the oracle. B has more rows because the lookahead tells apart states that look the same in features. Served as Q8_0 in Ollama, each chose the same option as its bf16 weights on 50 of 50 validation states.

**Agreement**, 1,000 held-out states. The oracle columns use section 11's states; the teacher column uses states drawn the same way from the teacher-labeled set.

| Model | With the teacher | With the oracle (decisive) | Cross-entropy vs oracle | Latency p50 |
|---|---|---|---|---|
| `tev1:0.8b` | 34.3% | 31.8% (29.5%) | 1.22 | 73 ms |
| `tev1:4b` | 38.2% | 38.4% (40.7%) | 1.29 | 205 ms |
| `pacman-0.8b` | 59.0% | 53.9% (57.5%) | 1.05 | 59 ms |
| C | 60.7% | 54.6% (57.3%) | 1.07 | 80 ms |
| A | 61.9% | 55.3% (59.3%) | 1.11 | 55 ms |
| B | 96.9% | 89.5% (96.1%) | 0.86 | 84 ms |

**In-game**, seeds 100-109, 1x, 5-minute cap:

| Player | Clock | Mean pellets | Mean score | Mean survival | Runs |
|---|---|---|---|---|---|
| `pacman-0.8b` | realtime | 386 | 6,348 | 81 s | 3 (456 from section 11; 300 and 400 rerun today) |
| C | realtime | 406 | 6,103 | 86 s | 1 |
| A | realtime | 425 | 7,866 | 86 s | 3 (441, 467, 366) |
| A | lockstep | 417 | 7,649 | 86 s | 1 |
| B | realtime | 765 | 11,865 | 191 s | 1 |
| B | lockstep | 1,387 | 22,713 | 298 s | 1 (9 of 10 games reached the cap) |
| Qwen3.8-27B with lookahead (teacher) | lockstep | 1,396 | 22,054 | 299 s | 1 |

Per game, B ate 463, 1,215, 889, 662, 469, 535, 698, 839, 1,199, and 676 pellets in real time, and 1,452, 1,329, 1,396, 1,452, 1,347, 1,401, 1,398, 1,391, 1,215, and 1,484 in lockstep.

**Generality**, the 194 JevBench items of section 15: B 0.61, A 0.53, `pacman-0.8b` 0.49, C 0.46, against 0.66 for `tev1:0.8b`.

What this shows:

- The teacher distills when the student sees what the teacher saw. B reproduces the teacher's choices on 96.9% of held-out states and its play in lockstep, at 85 ms per decision on the M3 Ultra instead of 309 ms on a 5090.
- With features only, the teacher's labels make no difference. A and C land on `pacman-0.8b`'s level: the run-to-run spread is larger than the gaps between them. The student's ceiling is its input, not its teacher (conclusion 8).
- All students side with the teacher more than with the oracle on the 104 states where the two disagree (`pacman-0.8b` 55% against 25%, B 84%). The sample is small.
- B loses 45% of its pellets in real time. It seldom misreads the lookahead: where one option's rollout died and another survived, it chose a dying one in 2.0% of such states, against 12.8% for Jev in section 16. The cause of the realtime loss was not isolated; its training states come from lockstep play, not from its own realtime games.
- Realtime results vary a lot between runs of the same model: `pacman-0.8b` scored 456, 300, and 400 in three 10-game runs. Today's runs shared the Mac with heavy unrelated CPU load (load average 15-36), so A and `pacman-0.8b` were alternated to face the same conditions. Compare realtime numbers within a day's runs, not across days.
- The lookahead rollouts run inside the encoder and their 5-10 ms of CPU is not charged to game time, as in section 16.

### 19. A hosted teacher instead of the 5090

Question: without the RTX 5090, what does labeling the 54,325 requests of section 18 cost through a hosted API, and does the setup carry over? The run used the `teacher-peek5s` prompt of section 18 (features plus 5 s rollout facts), thinking off through each API's own switch, temperature 0, and `gen-data --label-every`, which asks the teacher on every k-th decision of a full game. DeepSeek V4.1 Flash is not the section 18 teacher. Details, provider prices, and commands are in [guides/teacher-api.md](guides/teacher-api.md).

| Teacher (route) | States | Input / output tokens per request | Latency p50 | Failed answers | Same move as the section 18 labels | Projected cost, 54,325 requests |
|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash (Ollama Cloud) | 22 (greedy, seed 2000) | 455 / 31 | 0.60 s | 0 | 21 of 22 | $4.73 off-peak, $9.46 peak |

- Hosted APIs ignore the `chat_template_kwargs` switch the teacher sends to a local server, and hosted models can think by default, which spends output tokens before the answer. The teacher now takes `TEACHER_EXTRA_BODY` for each API's switch, plus `TEACHER_API_KEY` and retries on HTTP 429.
- Input and output contribute about equally to the bill: about 455 input tokens against 31 output tokens, at output prices 4-12 times input prices.
- Qwen3.8-27B itself is served by at least eight providers; at DeepSeek's measured token counts (an estimate for Qwen's tokenizer) the same run would cost about $6.90 at DeepInfra up to about $27 at Groq or Cerebras.
- The agreement column is a small sample (95% interval 78-99%) and shows that the prompt and answer format carry over, not that the model labels as well as the section 18 teacher. A student trained on another teacher's labels is a new model and needs its own evaluation.

### 20. Training the student on a Mac with MLX

Section 18 trained `pacman-0.8b-qwen` on one RTX 5090 in 32 minutes. This experiment retrains it on the Mac Studio that runs the games (Apple M3 Ultra, 80-core GPU, 512 GB, macOS 27.0) with MLX (`mlx` 0.32.3, `mlx-lm` 0.32.0), to measure the cost of the Mac path. The steps are in [guides/distill-mlx.md](guides/distill-mlx.md).

**Recipe.** `training/train_mlx.py` keeps section 18's student A: the same 31,570 training and 4,973 validation rows, the same prompt, the letter-only cross-entropy loss, LoRA rank 16 with scale 2.0 (alpha 32) on the same layers, AdamW at lr 1e-4 with warmup and cosine, gradient clipping at 1.0, batch 32, and the same row order. Each step runs as two micro-batches of 16 with averaged gradients. The run that was evaluated trained for one epoch (987 steps) instead of two.

**Three things had to change to get there.**

- `mlx-lm` 0.32 trains Qwen3.5's Gated DeltaNet layers with a loop over every token that keeps a float32 state per token for the backward pass. At batch 32 it was killed for memory; with 8-row micro-batches and gradient checkpointing it ran at 32 s per step, about 17.5 hours for two epochs. The script replaces it in training with the chunked form of the same recurrence (the algorithm of the torch fallback the 5090 ran), which matches `mlx-lm`'s loop to about 1e-7 in outputs and 1e-5 in gradients and runs at about 4 s per step. Writing it exposed a bug in `mlx` 0.32.3: the gradient of a step-2 slice over an axis of length 2 is wrong, silently.
- MLX's buffer cache grew by about 50 GB per step, because batches pad to different lengths, until macOS killed the process. A 4 GB cache limit holds memory flat.
- MLX's AdamW does not correct its moment estimates for their zero start unless asked (`bias_correction=False` by default); torch's always does. Without the correction the first few hundred updates are several times larger, and after 200 steps validation agreement was 39.8% against the 5090 run's 57.2%. With it, 53.5%.

**Training**, agreement with the teacher's label on the first 2,000 validation rows, as `train.py` logs it:

| Step | 0 | 200 | 400 | 600 | 800 | 1,000 | 1,200 | 1,400 | 1,600 | 1,800 | 1,974 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX 5090, 2 epochs | 30.8% | 57.2% | 56.6% | 57.6% | 57.6% | 58.3% | 60.2% | 60.9% | 61.0% | 61.7% | 61.9% |
| M3 Ultra, 2 epochs (stopped at step 1,600) | 31.3% | 53.5% | 57.5% | 59.6% | 57.3% | 59.6% | 59.7% | 59.3% | | | |
| M3 Ultra, 1 epoch | 31.3% | 55.3% | 56.6% | 58.9% | 59.4% | 59.6% (step 987) | | | | | |

The two-epoch Mac run was stopped by the tool that launched it, after 1,600 steps and before it saved an adapter; its curve had been flat since step 600, and the one-epoch run, whose cosine reaches zero at step 987, ended at the same level (59.6%, cross-entropy 0.979 against 0.966 for the 5090 run). The 5090 run gained about 3.6 points between steps 1,000 and 1,974, mostly as its learning rate fell; whether a full two-epoch Mac run would gain the same was not measured.

**Time and memory** (one-epoch run, Mac otherwise idle, load average 3-6):

| | M3 Ultra, MLX | RTX 5090, Unsloth |
|---|---|---|
| Steps | 987 (1 epoch) | 1,974 (2 epochs) |
| Seconds per step | 4.22 | 0.87 |
| Training loop, including validation checks | 74.5 min (5 checks, 5.1 min) | 31 min (10 checks) |
| Whole process, including model load, tokenizing, and the first check | 75.8 min | about 32 min |
| Prompt tokens per second | 2,807 | |
| Peak GPU memory | 47.8 GB (micro-batch 16) | |
| Merge and Q8_0 export | 14 s | |

The two-epoch Mac run ran at 4.13 s per step over its 1,600 steps (on a shared Mac, load average up to about 40), so two epochs would take about 2 hours 26 minutes. Smaller micro-batches with gradient checkpointing cut peak memory to 4.6 GB at about 2,350 tokens per second (3-step measurements in the guide).

**Serving.** Merged into the base checkpoint, exported to Q8_0 (795 MB), and imported into Ollama 0.35.1 as `pacman-0.8b-qwen-mlx`, it chose the same option as its MLX weights on 49 of 50 validation states (largest probability difference 0.04). Ollama 0.35.1 serves `/v1/systemone` only for models that declare the decision capability, and the published model, imported under 0.35.0, did not; for the comparison it was re-created from itself with `CAPABILITY decision` added, which changes no weights.

**Agreement**, 1,000 held-out states, served Q8_0, measured the same evening:

| Model | With the teacher | Cross-entropy | Latency p50 |
|---|---|---|---|
| `pacman-0.8b-qwen` (5090, 2 epochs) | 61.9% | 0.88 | 52 ms |
| MLX, 1 epoch | 59.4% | 0.91 | 52 ms |

**In-game**, seeds 100-109, 1x, 5-minute cap, the two models alternated in one session (load average 6-12):

| Player | Clock | Mean pellets | Mean score | Mean survival | Runs |
|---|---|---|---|---|---|
| `pacman-0.8b-qwen` | realtime | 427 | 7,820 | 86 s | 3 (366, 403, 511) |
| MLX, 1 epoch | realtime | 371 | 6,025 | 69 s | 3 (343, 413, 359) |
| `pacman-0.8b-qwen` | lockstep | 417 | 7,649 | 86 s | 1 |
| MLX, 1 epoch | lockstep | 352 | 5,949 | 70 s | 1 |

Decision p50 was 53 ms for both in real time. Paired by seed and run, the realtime difference is -55 pellets per game with a standard error of 34. Per game in lockstep, the MLX student ate 382, 217, 233, 243, 229, 447, 849, 240, 443, and 238 pellets; `pacman-0.8b-qwen` ate 468, 455, 233, 432, 212, 308, 416, 480, 439, and 730.

What this shows:

- The Mac can train this student in about an hour and a quarter per epoch, at about 5 times the 5090's time per step. It needs about 48 GB of GPU memory at the measured settings, or under 5 GB with gradient checkpointing at a further cost in speed.
- The training path needed three fixes beyond porting the loss: a chunked Gated DeltaNet recurrence, a cache limit, and Adam's bias correction. The last one changes results silently: the run still trains, only worse.
- The one-epoch student is a little weaker than the published two-epoch one: 2.5 points lower agreement with the teacher, 55 fewer pellets per realtime game (about 1.6 standard errors), and 65 fewer in lockstep. It plays at the level of `pacman-0.8b` (386) and student C (406) of section 18. The likely cause is the shorter schedule, since the 5090 run gained 3.6 points of validation agreement in its second epoch; the Mac two-epoch run showed no such gain through step 1,400, before its learning rate had fallen far, so this is not settled.
- The published model reproduced its section 18 numbers on Ollama 0.35.1 once the decision capability was declared: 61.9% agreement and 417 pellets in lockstep exactly, and 427 in real time against 425.

### 21. OpenAI's Decisions API with GPT-6 Luna

OpenAI announced the Decisions API at DevDay 2026 (2026-09-29) and opened it to all developers as a public beta on 2026-10-06. It is a dedicated endpoint, `POST /v1/decisions`, that runs GPT-6 Luna (`gpt-6-luna`, the only model it serves) on questions whose answers are fixed in advance: `predicate` (the probability that a condition holds), `choice` (one of the supplied values, with a probability for every value and a confidence), and `score` (a probability-weighted level). It generates no text. Input costs $0.10 per million tokens, and there is no output charge; replies report 0 output tokens. That is the shape of Jev's `/v1/systemone`, so it is a decision model in the sense this project uses, and the question is how it plays.

**Setup.** The policy `openai:gpt-6-luna` (`src/agent/decisions.ts`) sends the facts Jev gets: the `features` state as `input`, and one `choice` question named `move` whose instructions are the encoder's and whose choices are the legal options, with the encoder's descriptions ("turn around now" for `back`). The per-option probabilities are logged as for Jev. A refusal or a choice outside the options would count as a failed decision; none occurred. Both players ran over the internet from the M3 Ultra (load average about 6-8), on `features`, the realtime clock, 1x, seeds 100-109, and the 5-minute cap. There were two pairs of 10-game runs, and within each pair GPT-6 Luna and Jev ran at the same time, so they shared the network and the machine.

| Player | Run | Mean pellets | Mean survival | Decision p50 (mean of games) | p50 / p90 / p99, all decisions | Stale | Failed |
|---|---|---|---|---|---|---|---|
| GPT-6 Luna | 1 | 238.9 | 77 s | 177 ms | 164 / 255 / 772 ms | 6.6% | 0 of 3,328 |
| GPT-6 Luna | 2 | 234.6 | 68 s | 158 ms | 158 / 211 / 555 ms | 4.7% | 0 of 3,312 |
| Jev 1.13.0 | 1 | 187.8 | 57 s | 115 ms | 113 / 167 / 805 ms | 3.9% | 1 (HTTP 529, overloaded) |
| Jev 1.13.0 | 2 | 177.5 | 61 s | 113 ms | 113 / 154 / 241 ms | 2.7% | 0 of 3,960 |

Per game, GPT-6 Luna ate 243, 227, 196, 227, 484, 179, 240, 224, 229, and 140 pellets in run 1, and 219, 198, 222, 224, 228, 278, 330, 242, 167, and 238 in run 2. In run 1 on seed 104 it cleared the first level (244 pellets) and ate 240 on the second. One game (run 1, seed 101) ran at a p50 of 301 ms during a slow patch of the API; the other 19 games ran at 154-167 ms.

**Paired comparison** (GPT-6 Luna minus Jev, pellets per game, by seed):

| Pairing | Mean difference | Standard error | Difference / SE |
|---|---|---|---|
| Run 1, same session | +51.1 | 27.1 | 1.9 |
| Run 2, same session | +57.1 | 16.4 | 3.5 |
| Both runs, 20 paired games | +54.1 | 15.4 | 3.5 |
| Run 1 against section 12's Jev run (2026-09-30) | +61.1 | 36.5 | 1.7 |

GPT-6 Luna ate more than Jev in 18 of the 20 same-session pairs. Without run 1's seed 104, the mean difference is +42.1 with a standard error of 10.2. Jev reproduced its section 12 result: against that run, its first run of this session differs by +10.0 pellets per game with a standard error of 14.4.

**Tokens and cost.** The same state costs 359 input tokens on the Decisions API against about 609 on Jev, which adds about 300 tokens of its own (section 12). A 10-game run used 1.19 million input tokens, $0.12 at the published price, about $0.012 per game. The two runs and a 3-game smoke test cost about $0.28 together. Jev still answered as `jev-1.13.0`.

**Answers.** GPT-6 Luna's choices covered every direction (run 1: up 1,087, left 797, back 630, right 426, down 388). Its reported confidence averaged 0.33 against Jev's 0.44, and its top option's probability averaged 0.54 against 0.61. Three identical requests on one probe state returned 0.81, 0.79, and 0.82 for the top option, so the answers are not exactly deterministic.

**In the browser** (dev server, seed 100, 1x, through the `/openai` proxy), decisions took a p50 of about 250 ms and a p90 of about 400 ms, against about 210 ms for Jev measured the same way on 2026-10-05. The proxy hop costs the browser about 90 ms over the headless runner.

What this shows:

- GPT-6 Luna through the Decisions API plays at the level of the small chat models of section 14 (`phi4-mini` 232, `gemma4:e4b` 230) and above Jev, by about 54 pellets per game in the same sessions, at about 50 ms more per decision. It remains far below the distilled `pacman-0.8b-qwen` (425).
- The request maps one-to-one onto Jev's: state as evidence, one choice question, a probability per option. The same facts and the same question went to both, so the difference is the model behind the endpoint.
- Limits: two 10-game runs; a public beta whose model, latency, and price may change; the comparison with section 14's chat models is across days; the answers vary slightly between identical requests. Agreement with the teacher and JevBench accuracy were not measured.

### 22. Rebuilding the student on H100s, served by vLLM

Section 18 built `pacman-0.8b-qwen` with an NVFP4 teacher on an RTX 5090 and served it from Ollama. This rebuild ran the same pipeline on NVIDIA H100 80GB cards in a Kubernetes cluster and served the student with vLLM instead of Ollama, to check that the recipe reproduces on other hardware and another serving stack. The commands are in [guides/distill.md](guides/distill.md). The rebuilt student is called `pacman-0.8b-qwen-h100` here.

**Teacher.** Qwen3.8-27B in BF16, served by vLLM 0.31.0. The published labels came from NVFP4 weights, so the labels are not identical. On `scripts/probe_teacher.py` with thinking off, it scored 0.93 on 40 scenarios at concurrency 1 (p50 646 ms) and 0.96 on 80 at concurrency 16 (11.1 labels per second). The NVFP4 teacher scored 0.97 at 270 ms (section 6).

**Labels.** The same players and seeds as section 18, one `gen-data` process per game, 48 at a time, against two teacher replicas on two H100s. The 93 games took 41 minutes and sent 55,543 requests, with 468 input and 38 output tokens per request on average and 0 reasoning tokens. The longest game sent 1,275 requests, against 1,276 in the published run. 46 of 65,253 states were dropped unlabeled: 37 requests failed at the network level, and 9 answers had no JSON within 200 tokens. After deduplication there were 31,580 training rows and 4,975 validation rows, against 31,570 and 4,973.

**Training.** `training/train.py` unchanged, 2 epochs, one H100, at 0.74 s per step against 0.87 on the 5090. Agreement with the teacher's label on the first 2,000 validation rows, as `train.py` logs it (each run against its own teacher's labels):

| Step | 0 | 200 | 400 | 600 | 800 | 1,000 | 1,200 | 1,400 | 1,600 | 1,800 | 1,974 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX 5090 (section 20) | 30.8% | 57.2% | 56.6% | 57.6% | 57.6% | 58.3% | 60.2% | 60.9% | 61.0% | 61.7% | 61.9% |
| H100 | 30.9% | 55.7% | 58.1% | 59.0% | 58.9% | 60.2% | 60.2% | 60.7% | 62.5% | 62.0% | 62.4% |

The final validation cross-entropy was 0.960, against 0.966 for the 5090 run.

**Serving.** The merged bf16 weights, with no GGUF step, served by vLLM 0.31.0 behind `scripts/systemone_vllm.ts`. On 200 validation states, the served model and the bf16 weights scored with `training/score_hf.py` chose the same option on 197. The three that differ are ties or near-ties on the training side (margins 0, 0, and 0.046). Probabilities differed by at most 0.031, median 0.008. For the Ollama Q8_0 model of section 18 the figures were 48 of 50, 0.027, and 0.010.

The first full-model parity check found a bug in the adapter. vLLM returns the top log-probabilities of the whole vocabulary by default, even when `allowed_token_ids` restricts the output to the option letters. The fine-tuned model puts mass on letters past the last option: on one three-option state, A 0.49, C 0.14, D 0.12, B 0.11. Option B dropped out of the returned list, and the adapter scored it 0. Choices were unaffected, but cross-entropy against the teacher read 1.16 instead of 0.88. vLLM's `--logprobs-mode processed_logprobs` takes the log-probabilities after the restriction, and the adapter now refuses to answer when a letter is missing. A first parity check on the base model had passed because its top tokens were the letters anyway.

**Agreement**, 1,000 held-out states:

| Model | With its teacher | Cross-entropy | With the search | Latency p50 |
|---|---|---|---|---|
| `pacman-0.8b-qwen` (section 18; Ollama Q8_0 on the M3 Ultra) | 61.9% | 0.88 | 55.3% | 52 ms |
| `pacman-0.8b-qwen-h100` (vLLM bf16 on an H100, over the local network) | 61.4% | 0.88 | 56.1% | 37 ms |

Each student is scored against its own teacher's labels on the same validation states. The search column uses states from the validation seeds of section 18, labeled by `oracle-5s`. On `scripts/probe_decision_models.py` the rebuild scored 0.85 (13 of 13 frightened-ghost scenarios) at 35 ms.

**In-game**, seeds 100-109, 1x, 5-minute cap. The headless runner ran on a separate 8-core VM; the realtime runs went one after another with nothing else running:

| Player | Clock | Mean pellets | Mean score | Mean survival | Runs |
|---|---|---|---|---|---|
| `pacman-0.8b-qwen` (sections 18 and 20) | lockstep | 417 | 7,649 | 86 s | 1 |
| `pacman-0.8b-qwen-h100` | lockstep | 333 | 5,632 | 70 s | 2 (351, 316) |
| `pacman-0.8b-qwen` (sections 18 and 20) | realtime | 425 | | 86 s | 3 (441, 467, 366), M3 Ultra, 55 ms |
| `pacman-0.8b-qwen-h100` | realtime | 482 | 8,223 | 94 s | 3 (482.5, 481.2, 481.2), 37 ms |

Per game in lockstep, the rebuild ate 407, 235, 224, 456, 226, 394, 218, 469, 443, and 434 pellets with ten games in parallel, and 242, 235, 224, 242, 443, 239, 218, 423, 443, and 453 with one game at a time. Three seeds gave the same game both times; the rest diverged. Paired by seed with the published lockstep run, the differences are -67 (standard error 40) and -101 (standard error 48) pellets per game. In real time, runs 2 and 3 were identical game for game, and run 1 differed on four seeds.

What this shows:

- The pipeline reproduces on H100s with vLLM in place of the 5090 and Ollama. The data has the same size, the training curve ends 0.5 points higher, and offline agreement with the teacher (61.4% against 61.9%), cross-entropy (0.88), and agreement with the search (56.1% against 55.3%) match.
- vLLM serves the bf16 weights as faithfully as Ollama served Q8_0, provided `--logprobs-mode processed_logprobs` is set. Without it the probabilities are wrong while the choices look right, so a check on choices alone would have missed the bug.
- The in-game result is not settled. The rebuild ate fewer pellets than the published model in lockstep (333 against 417, about 1.7 and 2.1 standard errors in the two runs) and more in real time (482 against 425). The realtime runs used other hardware at lower latency, so they do not compare across sections. Lockstep games are not reproducible from run to run: the parallel and sequential runs diverged on 7 of 10 seeds. The cause is a guess: vLLM's batch-dependent bf16 numerics, which flip near-ties. A same-session control with the published weights on vLLM was not possible: its GGUF uses the `qwen35` architecture, which transformers 5.17 cannot load from GGUF.

## Conclusions

1. Use `tev1:4b` for the real-time demo. At about 160 ms, a decision is shorter than the typical 400 ms to 1.2 s Pac-Man needs to reach the next junction at arcade speed.
2. Use a 0.8B model as the fine-tuning target and `tev1:0.8b` as its off-the-shelf baseline. The fine-tuned `pacman-0.8b` beats every off-the-shelf model and the scripted rule in play, at `tev1:0.8b`'s latency.
3. Label training data with `oracle-5s`. It is far stronger than any model we tried and costs milliseconds of local CPU per label. It also scores every option, so labels can be soft: ties become visible instead of being broken arbitrarily. Keep Qwen3.8-27B only as a comparison point.
4. Mind the information gap. The oracle sees the full game state, and the student sees only its encoding. Decisions that depend on facts the encoding drops cannot be learned from it. Measure the student against the oracle on held-out states to see how large this gap is.
5. Invest in the encoder before the model. Keep states compact and factual, and measure every encoder change in tokens as well as accuracy.
6. Compare clocks before comparing players. All Qwen results are lockstep, where the game waits for each answer; Jev, the Tev1 models, and `pacman-0.8b` are realtime, where slow answers arrive after the junction has passed. Qwen without thinking (301 pellets) beating Jev (178) does not yet show it would at realtime.
7. Do not argue cost against Jev. At the price seen on Vercel AI Gateway ($0.042 per million input tokens), a whole game of about 340 decisions at 605 tokens each costs under a cent. The case for a fine-tuned local model is task accuracy, latency, offline use, and control. The cost case holds against calling a large general model at every step.
8. A stronger teacher may not make a stronger student. The student's ceiling is set by what it sees at inference time. Qwen with lookahead beats the oracle by reading rollouts the features-only student never sees, so relabeling with it may leave the student near its current plateau. A student given the same lookahead facts would test this; the rollouts take milliseconds of CPU. Section 18 confirms it: relabeling with the Qwen teacher did not change the features-only student, while a student given the lookahead matched the teacher.
9. Treat a plain chat model with constrained JSON as the off-the-shelf baseline. On this task it matches or beats the decision models (Jev, `tev1:4b`) from the same facts, so the decision API is a convenience here, not a capability. `phi4-mini` is the fastest and strongest of those tried and returns probabilities. The fine-tuned `pacman-0.8b` still doubles it.
10. Keep lookahead out of inference. A rollout of the real engine is close to seeing the future, so it may label training data but must not be part of any player's input when players are compared. The distilled 0.8B to cite is `pacman-0.8b-qwen` (features input), not the lookahead student.

## Possible follow-ups (not planned)

The project stops here: the demo sequence (off-the-shelf 0.8B, off-the-shelf 4B, fine-tuned 0.8B) is complete. If it resumes, the gap to the oracle is the obvious target:

- Enrich the encoder: ghost identities, the second-nearest ghost, and longer-range pellet counts.
- Run a DAgger round with states from `pacman-0.8b`'s own games.
- Try a full-map student: fine-tune on `ascii-full` with oracle labels and compare.
