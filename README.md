# Decision Pac-Man

A model steers a Pac-Man-style game in the browser in real time at 30 ticks per second, choosing among the legal moves at each junction. The repository compares off-the-shelf models (the decision models Jev, Tev1, and GPT-6 Luna through OpenAI's Decisions API, and plain chat models answering in constrained JSON) with a 0.8B model distilled from a slow general LLM teacher. It contains the browser game, a headless runner for seeded games, data generation and fine-tuning scripts, evaluation benchmarks, and an iPhone app. The distilled model is [on Hugging Face](https://huggingface.co/grapeot/decision-pacman-0.8b-GGUF).

## Results

Every player sees the same current-state input (`features`), with no lookahead. Each row is 10 games on evaluation seeds 100-109, at 1x speed with a 5-minute cap, on the realtime clock: a slow answer costs game time, as it does in the browser.

| Player | Mean pellets | Decision p50 |
|---|---|---|
| random | 30 | 0 ms |
| `tev1:0.8b` | 69 | 62 ms |
| `tev1:4b` | 148 | 202 ms |
| Jev 1.13.0 (hosted) | 178 | 115 ms |
| `llm:qwen3.5:4b` | 197 | 267 ms |
| greedy (scripted) | 201 | 0 ms |
| `llm:gemma4:e4b` | 230 | 253 ms |
| `llm:phi4-mini` | 232 | 187 ms |
| `openai:gpt-6-luna` (OpenAI Decisions API, hosted) | 237, mean of 2 runs (239, 235) | 168 ms |
| `pacman-0.8b-qwen` (0.8B distilled from Qwen3.8-27B; [weights](https://huggingface.co/grapeot/decision-pacman-0.8b-GGUF)) | 425, mean of 3 runs (441, 467, 366) | 55 ms |

Lockstep results are listed apart, because there the game waits for each answer and the numbers are not comparable with the ladder: Qwen3.8-27B scored 301 pellets with thinking off and 531 with thinking on (decision p50 2.4 s), both over 3 games on seeds 100-102; `pacman-0.8b-qwen` scored 417 over seeds 100-109.

Full numbers, caveats (clock effects, run-to-run variance up to ~150 pellets, hardware), and the lookahead rule are in [docs/results.md](docs/results.md).

## What we found

- A 0.8B model (LoRA on the Qwen3.5-0.8B base) distilled from Qwen3.8-27B averaged 425 pellets (3-run mean) at 55 ms per decision, against 69 for the same-size off-the-shelf `tev1:0.8b` and 178 for Jev. The teacher chose each move while seeing a simulated 5-second future per option, but only while labeling the training data; the student reads only the current state.
- Stock small chat models with constrained JSON (`llm:phi4-mini` 232, `llm:gemma4:e4b` 230, `llm:qwen3.5:4b` 197) matched or beat the decision models `tev1:4b` (148) and Jev (178) from the same facts.
- OpenAI's Decisions API with GPT-6 Luna, asked the same question in the same request shape as Jev, averaged 237 pellets at 168 ms. Run at the same time as Jev on the same seeds, it ate 54 more pellets per game (standard error 15, 20 paired games), with no failed or illegal answers, at about $0.012 per game.
- Specialization cost generality: on 194 JevBench items, `pacman-0.8b-qwen` scored 0.53 accuracy against 0.66 for `tev1:0.8b` (chance 0.32).
- Decisions are prefill-bound (`docs/model_evaluation.md` section 17): decision models read the answer directly from option logits, whereas chat models writing ~6-token JSON pay about 72 ms more on an Apple M3 Ultra and 27 ms on an RTX 5090.
- Rebuilding the student on H100s with Qwen3.8-27B BF16 and serving it through vLLM (`scripts/systemone_vllm.ts` with `--logprobs-mode processed_logprobs`) matched the published model within noise (-3.9 pellets per game over 50 paired lockstep seeds, standard error 29.9). Swapping the teacher for DeepSeek-V4-Flash (91.1% label agreement with Qwen), training on Qwen's token logprobs (`TEACHER_LOGPROBS=1`), or averaging both teachers stayed within 1.1 to 1.3 standard errors over up to 100 evaluation seeds (`docs/model_evaluation.md` sections 22-24).

## Watch

30-second clips, all on seed 100 at 1x speed:

| Clip | Player | Pellets | Lives lost |
|---|---|---|---|
| [`docs/media/demo_tev1_08b.mp4`](docs/media/demo_tev1_08b.mp4) | `tev1:0.8b` | 71 | 2 |
| [`docs/media/demo_tev1_4b.mp4`](docs/media/demo_tev1_4b.mp4) | `tev1:4b` | 101 | 1 |
| [`docs/media/demo_jev.mp4`](docs/media/demo_jev.mp4) | Jev 1.13.0 (hosted) | 86 | 1 |
| [`docs/media/demo_qwen35_4b.mp4`](docs/media/demo_qwen35_4b.mp4) | `llm:qwen3.5:4b` | 109 | 0 |
| [`docs/media/demo_phi4_mini.mp4`](docs/media/demo_phi4_mini.mp4) | `llm:phi4-mini` | 143 | 2 |
| [`docs/media/demo_pacman_08b_qwen.mp4`](docs/media/demo_pacman_08b_qwen.mp4) | `pacman-0.8b-qwen` | 176 | 0 |

## Quickstart

Requirements: Node 20+, Ollama 0.35+.

```bash
ollama pull tev1:4b
npm install
npm run dev            # open http://localhost:5173
```

Switch models, encoders, and speed in the side panel, or pass URL parameters such as `?model=tev1:4b`.

To play the distilled 0.8B, download it from [Hugging Face](https://huggingface.co/grapeot/decision-pacman-0.8b-GGUF) and import it into Ollama. No GPU is needed; the `hf` command comes with `huggingface_hub` (`pip install -U huggingface_hub`).

```bash
hf download grapeot/decision-pacman-0.8b-GGUF pacman-0.8b-qwen-Q8_0.gguf Modelfile.pacman-0.8b-qwen --local-dir .
echo 'CAPABILITY decision' >> Modelfile.pacman-0.8b-qwen
ollama create pacman-0.8b-qwen -f Modelfile.pacman-0.8b-qwen
```

Then open `http://localhost:5173/?model=pacman-0.8b-qwen`. The GGUF (Q8_0, 795 MB) is the evaluated model, and the Modelfile carries the system prompt it was trained under. Ollama copies the weights into its own store, so the downloaded files can be deleted afterwards. Ollama 0.35.1 and later serve `/v1/systemone` only for models that declare `CAPABILITY decision`, which the published Modelfile lacks (Ollama 0.35.0 rejected the line); the `echo` adds it. On 0.35.0, skip the `echo`. If you imported the model before upgrading Ollama and it now answers "does not support decision", add the capability in place: `printf 'FROM pacman-0.8b-qwen\nCAPABILITY decision\n' > Modelfile.decision && ollama create pacman-0.8b-qwen -f Modelfile.decision`.

## iPhone

`ios/` runs the same game on an iPhone with three players: the distilled 0.8B and phi4-mini on device, and Jev in the cloud. See [docs/guides/iphone.md](docs/guides/iphone.md).

[`docs/media/iphone_pacman_08b_qwen.mp4`](docs/media/iphone_pacman_08b_qwen.mp4) is a 29-second screen recording of the distilled `pacman-0.8b-qwen` on an iPhone 16 Pro Max at 1x: about 505 ms per decision, 25-28% late answers, no lives lost.

## Reproduce each result

| Result | Command | Guide |
|---|---|---|
| Random and greedy baselines | `npm run headless -- --policy random --games 10 --seed 100 --max-seconds 300`<br>`npm run headless -- --policy greedy --games 10 --seed 100 --max-seconds 300` | [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) section 11 |
| `tev1:0.8b` and `tev1:4b` | `npm run headless -- --policy tev1:0.8b --games 10 --seed 100 --max-seconds 300`<br>`npm run headless -- --policy tev1:4b --games 10 --seed 100 --max-seconds 300` | [docs/guides/run-models.md](docs/guides/run-models.md), [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) section 11 |
| Jev (hosted) | `npm run headless -- --policy jev --games 10 --seed 100 --max-seconds 300`<br>(with `TYPESAFE_API_KEY` set; take the key from your password manager) | [docs/guides/run-models.md](docs/guides/run-models.md), [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) sections 11, 12 |
| GPT-6 Luna (OpenAI Decisions API) | `npm run headless -- --policy openai:gpt-6-luna --games 10 --seed 100 --max-seconds 300`<br>(with `OPENAI_API_KEY` set) | [docs/guides/run-models.md](docs/guides/run-models.md), [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) section 21 |
| Plain chat models (`llm:phi4-mini` etc.) | `ollama pull phi4-mini`<br>`npm run headless -- --policy llm:phi4-mini --games 10 --seed 100 --max-seconds 300` | [docs/guides/run-models.md](docs/guides/run-models.md), [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) section 14 |
| `pacman-0.8b-qwen` (download it as in the Quickstart, or train it yourself) | `npm run headless -- --policy pacman-0.8b-qwen --games 10 --seed 100 --max-seconds 300` | [docs/guides/run-models.md](docs/guides/run-models.md), [docs/guides/distill.md](docs/guides/distill.md), [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) section 18 |
| Lockstep rows | `npm run headless -- --policy teacher --clock lockstep --games 3 --seed 100 --max-seconds 300`<br>`npm run headless -- --policy teacher-think --clock lockstep --games 3 --seed 100 --max-seconds 300`<br>`npm run headless -- --policy pacman-0.8b-qwen --clock lockstep --games 10 --seed 100 --max-seconds 300`<br>(the teacher needs `TEACHER_BASE_URL` and `TEACHER_MODEL` in `.env`) | [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) sections 10, 13, 18 |
| Agreement with the teacher | `python3 training/score_ollama.py --model <name> --data data/q1/val --target label --n 1000` | [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) section 18 |
| Generality | `python3 scripts/eval_general_decisions.py tev1:4b tev1:0.8b` | [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) sections 15, 18 |
| Latency breakdown | `python3 scripts/bench_prefill_decode.py --backend ollama --model tev1:4b` | [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) section 17 |
| Clips | `npm run record -- --model tev1:4b --seconds 30 --seed 100 --out docs/media/demo.mp4` | [docs/guides/run-models.md](docs/guides/run-models.md) |

## Guides

- [docs/guides/run-models.md](docs/guides/run-models.md): Play any model in the browser or headless runner.
- [docs/guides/evaluate.md](docs/guides/evaluate.md): Evaluation protocol, seeds, clocks, variance, and scoring scripts.
- [docs/guides/distill.md](docs/guides/distill.md): Label, train, export, import, and score a fine-tuned model (requires a CUDA GPU).
- [docs/guides/teacher-api.md](docs/guides/teacher-api.md): Label states with a hosted Qwen3.8-27B instead of a local GPU: providers, prices, settings, and measured cost.
- [docs/guides/distill-mlx.md](docs/guides/distill-mlx.md): Train and export the same model on an Apple-silicon Mac with MLX, with measured time and memory.
- [docs/guides/iphone.md](docs/guides/iphone.md): Build and run the native iOS app with three selectable players.
- [docs/results.md](docs/results.md): Canonical current numbers, hardware details, caveats, and the lookahead rule.
- [docs/model_evaluation.md](docs/model_evaluation.md): Chronological lab notebook across experiments 1 through 24.
- [docs/rfc.md](docs/rfc.md): Architecture and technical decisions.
- [docs/prd.md](docs/prd.md): Original requirements.
- [docs/working.md](docs/working.md): Changelog and development notes.

## Tests

```bash
npm test
```

## FAQ

### Why not a hand-written heuristic?

One is on the ladder: the scripted `greedy` rule scored 201 pellets. A much stronger one is possible here because the game has an exact simulator (ghost moves are deterministic except frightened ghosts' random turns). A search that copies the game and rolls the engine forward 5 seconds per option (`oracle-5s`, in `src/agent/oracle.ts`) scores 1,109 pellets per game at about 5 ms of CPU per decision, on the ladder's seeds, clock, and cap. It is not on the ladder because it sees the future at decision time, and lookahead may label training data but never enters a player's input in a comparison. Real tasks rarely have a simulator to search with, so a general LLM teacher is the recipe that transfers. An earlier 0.8B (`pacman-0.8b`) distilled from that search's labels scored at the same level as the LLM-distilled one: 386 pellets (mean of three runs: 456, 300, 400) against 425, a gap smaller than the run-to-run spread of up to about 150 pellets.

The numbers are collected in [docs/results.md](docs/results.md#faq-the-search-oracle-and-the-first-student); details are in [docs/model_evaluation.md](docs/model_evaluation.md), sections 10, 11, and 18.

## License

MIT. Pac-Man is a trademark of Bandai Namco. This project is not affiliated with it and uses no original art or audio.
