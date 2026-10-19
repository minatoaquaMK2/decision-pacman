# Test strategy

## Unit tests (Vitest, offline, default `npm test`)

- Engine: movement and wall collision, buffered intent (turn at the first legal tile, immediate reverse), tunnel wrap, pellet and power pellet scoring, frightened timer, ghost eaten and death transitions, level clear.
- Ghost AI: each ghost's target tile in scatter and chase for fixed positions, and no reversal except on mode change.
- Determinism: the same seed and intent sequence produce identical state after N ticks.
- Decision points: next-junction detection and the exit set for every junction on the maze.
- Encoders: output for fixed states is snapshot-tested, the options match the legal exits, the serialized size stays under the token budget (character count as a proxy), and no verdict fields (safe, danger, recommend) appear.
- Agent loop: with a fake client, at most one request is in flight, an answer for a stale junction is dropped, and a failed request leaves the intent unchanged.
- vLLM adapter (`scripts/systemone_vllm.ts`), against a fake vLLM server: the answer is the softmax over the option letters, keyed by option; each request asks for one token restricted to the option letters on the prompt Ollama scores; a letter missing from vLLM's log-probabilities is an error, never a zero; an unknown model is a 404.

## Integration tests (opt-in, need a running Ollama)

- Planned, not implemented yet: `npm run test:live`, skipped unless `LIVE_DECISION_API=1`. It sends one request per encoder to each configured model and checks the response schema, that the chosen option is in the option set, and that the probabilities sum to about 1.
- `python3 scripts/probe_decision_models.py <model>`: prints latency, input tokens, and accuracy on 40 synthetic scenarios per model. Record the results in `working.md` when models, encoders, or hardware change.

## Headless evaluation (manual, needs Ollama)

The full protocol is in [guides/evaluate.md](guides/evaluate.md).

`npm run headless -- --policy tev1:4b --encoder features --clock realtime --games 10 --seed 100 --max-seconds 300` plays full games and writes JSONL under `runs/` (gitignored). A change to an encoder or to the agent loop is done when its summary is compared against the previous run and the `random` and `greedy` baselines on the same seeds, and the comparison is written to `working.md`.

## Manual browser check

With `npm run dev` and Ollama running: the tick rate reads 30, probability bars update, latency p50 matches the headless numbers for the same model and encoder, switching encoders mid-game works, and stopping Ollama mid-game shows the error panel while the game keeps running.

## Teacher data and fine-tuning

- Teacher adapter: offline tests with recorded responses cover parsing, illegal moves, and unparseable output (the state is dropped, never guessed).
- Dataset export: a schema check on every line, plus a check that no evaluation seed appears in train or validation.
- Prompt parity: before training, a script renders sample requests with the training formatter and compares them to the prompt the serving runtime builds. They must match exactly.
- Export smoke test: an untrained 0.8B export is served over `/v1/systemone` and answers the probe before any real training run.
- Fine-tuned model: `scripts/probe_decision_models.py` plus the held-out realtime games in `docs/model_evaluation.md`.

## iOS app

- `ios/scripts/mac_check.sh decision <gguf>` runs the app's decision engine on the Mac over `ios/mac_check/bench_prompts.json` and compares with Ollama's probabilities (same choice on all, max difference about 0.0003).
- `ios/scripts/mac_check.sh chat <gguf>` runs the chat engine over `ios/mac_check/chat_prompts.json`: every answer must be legal, and with phi4-mini's default context the moves and prompt token counts must match Ollama's recorded answers.
- Compile for device without signing: `xcodebuild -project ios/DecisionPacman.xcodeproj -scheme DecisionPacman -destination "generic/platform=iOS" CODE_SIGNING_ALLOWED=NO build` (after `ios/scripts/build_web.sh` and `xcodegen generate` in `ios/`). The llama.cpp xcframework has no simulator slice.
- On the phone: switch through all three players, read `Documents/game_status.json` for the active player, its load state, latency, and frame timing, and check that the previous model was freed (no memory warning or termination when switching to phi4-mini).

## E2E

None in v1. The canvas output is checked by hand. A Playwright smoke test (page loads, the game advances in human mode, the error panel appears without Ollama) is worth adding once the UI settles.
