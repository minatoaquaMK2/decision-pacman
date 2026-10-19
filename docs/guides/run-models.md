# Run models

A model can play the game in four ways: as a local decision model over Ollama's `/v1/systemone` API, as a plain chat model through Ollama's `/api/chat` (policy `llm:<model>`), as TypeSafe's hosted decision model Jev, or through OpenAI's Decisions API with GPT-6 Luna (policy `openai:gpt-6-luna`). The scripted baselines `random` and `greedy` need no model.

Requirements: Node 20+ with npm, and Ollama. Decision models need Ollama 0.35+, which added `/v1/systemone`.

Results are in [results.md](../results.md). The evaluation protocol is in [evaluate.md](evaluate.md).

## Local decision models (Ollama /v1/systemone)

Decision models served on Ollama's `/v1/systemone` endpoint output a probability for every option, read directly from option-letter logits.

Pull the public decision models:
```bash
ollama pull tev1:4b
ollama pull tev1:0.8b
```

Install dependencies and start the local development server:
```bash
npm install
npm run dev
```

Open `http://localhost:5173`. To select a specific model, specify the `model` URL parameter:
```
http://localhost:5173/?model=tev1:0.8b
```

The development server forwards `/decide` to `VITE_DECISION_BASE_URL` (default `http://localhost:11434`), avoiding browser CORS restrictions. To point the server to another endpoint, such as ollaya on port 11435, set the variable when starting the server:
```bash
VITE_DECISION_BASE_URL=http://localhost:11435 npm run dev
```

To run headless games from the command line:
```bash
npm run headless -- --policy tev1:4b --games 3 --seed 100
```

The headless runner writes outputs to `runs/<tag>/`:
- `decisions.jsonl`: One record per decision.
- `summary.json`: Aggregate statistics.

### The distilled 0.8B (`pacman-0.8b-qwen`)

The distilled model is published on Hugging Face at [grapeot/decision-pacman-0.8b-GGUF](https://huggingface.co/grapeot/decision-pacman-0.8b-GGUF). It runs on the same Ollama as the other decision models, with no GPU. Download the GGUF and its Modelfile, and import them:
```bash
hf download grapeot/decision-pacman-0.8b-GGUF pacman-0.8b-qwen-Q8_0.gguf Modelfile.pacman-0.8b-qwen --local-dir .
echo 'CAPABILITY decision' >> Modelfile.pacman-0.8b-qwen
ollama create pacman-0.8b-qwen -f Modelfile.pacman-0.8b-qwen
```
The `hf` command comes with `huggingface_hub` (`pip install -U huggingface_hub`). `pacman-0.8b-qwen-Q8_0.gguf` (795 MB) is the evaluated model. The Modelfile names the GGUF in `FROM`, sets `TEMPLATE {{ .Prompt }}`, and carries the system prompt from `training/prompt.py`, the one the model was trained under. Imported this way, it gave the same choices and probabilities as the evaluated model on 100 validation states. Ollama copies the weights into its own store, so the downloaded files can be deleted afterwards. Ollama 0.35.1 and later serve `/v1/systemone` only for models that declare `CAPABILITY decision`, which the published Modelfile lacks (Ollama 0.35.0 rejected the line); the `echo` adds it. On 0.35.0, skip the `echo`. If you imported the model before upgrading Ollama and it now answers "does not support decision", add the capability in place: `printf 'FROM pacman-0.8b-qwen\nCAPABILITY decision\n' > Modelfile.decision && ollama create pacman-0.8b-qwen -f Modelfile.decision`.

Run it by model name:
```bash
npm run headless -- --policy pacman-0.8b-qwen --games 3 --seed 100
```
In the browser, open `http://localhost:5173/?model=pacman-0.8b-qwen`. To train it yourself, see [distill.md](distill.md).

### Decision models served by vLLM

vLLM has no `/v1/systemone` endpoint. `scripts/systemone_vllm.ts` serves one in front of a vLLM server, so the browser game, the headless runner, and the scoring scripts play a model that vLLM serves. Serve the model under the name the clients ask for, then start the adapter on Ollama's port:
```bash
vllm serve runs/ft_q1/merged --served-model-name pacman-0.8b-qwen --logprobs-mode processed_logprobs --max-model-len 4096 --limit-mm-per-prompt '{"image":0,"video":0}'
npm run systemone-vllm -- --vllm http://localhost:8000 --port 11434
npm run headless -- --policy pacman-0.8b-qwen --games 3 --seed 100
```
If Ollama already uses port 11434, pick another port and pass it as `--base-url` (headless, `score_ollama.py`) or `VITE_DECISION_BASE_URL` (dev server). The adapter listens on 127.0.0.1 unless `--host` says otherwise.

For each choice question, the adapter renders the prompt Ollama scores (`src/agent/prompt.ts`), asks vLLM's `/v1/completions` for one token restricted to the option letters (`allowed_token_ids`), and renormalizes the letters' log-probabilities. That is the softmax over option-letter logits that Ollama returns and `training/train.py` trains. `--logprobs-mode processed_logprobs` is required: without it, vLLM returns the top tokens of the whole vocabulary, where letters past the last option can push an option out of the list, and the adapter answers with an error naming the missing letters. The prompt uses the Qwen3.5 chat template, so the adapter fits Qwen3.5-based models such as `pacman-0.8b-qwen`. `runs/ft_q1/merged` is the output of `training/merge.py` (see [distill.md](distill.md#serve-with-vllm-instead-of-ollama)).

## Plain chat models (llm:<model>)

Plain chat models run through Ollama's `/api/chat` endpoint using the policy format `llm:<ollama-model>`. The agent supplies the same facts as decision models, applies a JSON schema restricting `move` to legal options, turns thinking off, sets temperature to 0, and derives probabilities from token logprobs. Ollama 0.35.0 provides no logprobs for `gemma4:e4b`. The adapter code is in `src/agent/llm.ts`.

Pull a chat model:
```bash
ollama pull phi4-mini
```
The other chat models on the ladder are `gemma4:e4b` and `qwen3.5:4b`.

In the browser, open `http://localhost:5173/?model=llm:phi4-mini`. This works when the development server points to Ollama.

To run headless games:
```bash
npm run headless -- --policy llm:phi4-mini --games 3 --seed 100
```

## Hosted Jev

Jev is TypeSafe's hosted decision model at `https://api.typesafe.ai/v1/systemone`. It uses the same request and response format as Ollama's `/v1/systemone` endpoint. The model name is `jev-latest` (it answered as `jev-1.13.0`).

Take the API key from your password manager and set it as `TYPESAFE_API_KEY`. The hosted API rejects Ollama's `keep_alive` field, so hosted requests omit it.

To run headless games:
```bash
export TYPESAFE_API_KEY="..."   # paste from your password manager
npm run headless -- --policy jev --games 3 --seed 100
```
The headless runner also reads a local `.env` (gitignored; `.env.example` lists the variable), but exporting the key in the shell keeps it out of files. The headless policy also accepts `JEV_BASE_URL` and `JEV_MODEL`.

To run in the browser, export the key in your shell before starting the development server:
```bash
export TYPESAFE_API_KEY="..."   # paste from your password manager
VITE_DECISION_BASE_URL=https://api.typesafe.ai npm run dev
```
Open `http://localhost:5173/?model=jev-latest`. The development server reads the key from the process environment (not `.env`) and adds the key to proxied requests. The key never reaches the browser page.

## OpenAI Decisions API (GPT-6 Luna)

OpenAI's Decisions API (`POST https://api.openai.com/v1/decisions`, public beta since 2026-10-06) runs GPT-6 Luna on questions whose answers are fixed in advance. `gpt-6-luna` is the only model it serves. Like Jev, it generates no text: a `choice` question returns one of the supplied values, a probability for each value, and a confidence. Input costs $0.10 per million tokens, with no output charge.

The policy `openai:gpt-6-luna` (`src/agent/decisions.ts`) sends the same facts as the other players: the encoded state as `input`, and one `choice` question named `move` with the encoder's instructions and the legal options as its choices. The option probabilities are logged like Jev's. A refusal or a choice outside the legal options counts as a failed decision.

Create an API key in the OpenAI platform and set it as `OPENAI_API_KEY`, in the shell or in the gitignored `.env`.

To run headless games:
```bash
export OPENAI_API_KEY="..."   # paste from your password manager
npm run headless -- --policy openai:gpt-6-luna --games 3 --seed 100
```

To run in the browser, export the key in your shell before starting the development server, then open `http://localhost:5173/?model=openai:gpt-6-luna`:
```bash
export OPENAI_API_KEY="..."
npm run dev
```
The development server forwards `/openai` to `https://api.openai.com` and adds the key from the process environment (not `.env`), so the key never reaches the page. The browser path adds a proxy hop: decisions took about 250 ms there against about 160 ms headless.

## Baselines and labeling players

The repository provides scripted baselines, engine rollouts, and teacher policies:
- `random`: Selects uniformly at random among legal options.
- `greedy`: Flees nearby ghosts, chases edible ghosts, and otherwise heads for the nearest pellet.
- `oracle-5s` (and variants `oracle-3s`, `oracle-8s`): A search over engine rollouts (`src/agent/oracle.ts`). It sees the future at decision time, so it drives games during data generation but is not a ladder player; see the FAQ in the [README](../../README.md#faq).
- Chat teacher policies (`teacher`, `teacher-think`, `teacher-peek5s`): Connect to an OpenAI-compatible endpoint specified by `TEACHER_BASE_URL` and `TEACHER_MODEL` in `.env`. The `teacher-peek5s` policy adds a simulated 5-second engine rollout per option to the prompt (lookahead); it labels the training data for `pacman-0.8b-qwen`.

## Encoders

Encoders determine the state representation supplied to the player. Set an encoder with `--encoder` in headless runs or `?encoder=` in the browser:
- `features` (default): Compact JSON containing per-option facts (BFS distance to nearest ghost through that exit and whether it is frightened or approaching, pellets within N steps, distance to nearest power pellet, dead end within N steps) and global facts (frightened time left, pellets left, lives). It reports facts, never verdicts such as "safe" or a recommended move.
- `ascii-window`: A board crop around Pac-Man, for comparison only.
- `ascii-full`: The whole board with row and column numbers, for comparison only.
- `features-peek5s`: Extends `features` with the outcome of a 5-second engine rollout per option (seconds until death or null, pellets, power pellets, ghosts eaten, points). This encoder is lookahead. Use it for labels and records, never for a comparison between players.

## Browser controls

The side panel switches the model, the encoder, and the game speed (0.25x to 1.25x), or hands control to the keyboard (arrow keys, WASD, or swipes; Space pauses, R restarts, M mutes). The same settings can be set in the URL:

| Parameter | Meaning |
|---|---|
| `model` | Model or policy name (default `tev1:4b`) |
| `encoder` | Encoder name (default `features`) |
| `seed` | Game seed (random if omitted) |
| `speed` | Game speed multiplier (default 1) |
| `mode` | `ai` or `human` |
| `wait` | `1` turns on the Wait switch |

### Wait for the model

The "Wait" checkbox (or `?wait=1`) is for slow models. The model is asked about the next junction while the game keeps moving. If Pac-Man is about to reach that junction before the answer arrives, the whole game, ghosts and timers included, holds until it does. A slow model is then judged on its choices, not its speed. With Wait on, latency no longer costs game time, so the play you see is not comparable with the realtime ladder in [results.md](../results.md). The headless runner has no Wait switch; its `--clock lockstep` waits for every answer instead.


To record a 30-second gameplay video clip:
```bash
npm run record -- --model tev1:4b --seconds 30 --seed 100 --out docs/media/demo.mp4
```
The recording script requires Playwright's Chromium and ffmpeg. It accepts `--model`, `--encoder`, `--seed`, `--speed`, `--seconds`, and `--out`.

## Headless runner flags

Run games from the command line using `scripts/run_headless.ts`:
```bash
npm run headless -- [flags]
```

The runner logs per-game results, prints an aggregate summary line (`meanPellets`, `meanScore`, `meanSeconds`, `meanLatencyP50`, `staleRate`), and writes `runs/<tag>/decisions.jsonl` and `runs/<tag>/summary.json`. The runner reads `.env`.

| Flag | Default | Description |
|---|---|---|
| `--policy` | `tev1:4b` | Player policy (`tev1:4b`, `tev1:0.8b`, `pacman-0.8b-qwen`, `random`, `greedy`, `jev`, `openai:gpt-6-luna`, `llm:<model>`, `teacher`, etc.) |
| `--base-url` | `VITE_DECISION_BASE_URL`, else `http://localhost:11434` | Decision endpoint for model policies |
| `--encoder` | `features` | State encoder (`features`, `ascii-window`, `ascii-full`, `features-peek5s`) |
| `--clock` | `realtime` | Clock mode: `realtime` charges decision latency in elapsed ticks; `lockstep` pauses the game for each answer |
| `--games` | `3` | Number of games to run |
| `--seed` | `100` | Starting seed (game `i` uses `seed + i`) |
| `--speed` | `1` | Simulation speed multiplier |
| `--max-seconds` | `180` | Cap on game seconds per game |
| `--out` | `runs` | Output directory for run logs |

To execute the standard ladder benchmark (10 games, evaluation seeds 100-109, 5-minute cap per game):
```bash
npm run headless -- --policy tev1:4b --games 10 --seed 100 --max-seconds 300
```
For lockstep execution, pass `--clock lockstep`.

For full ladder scores and latency benchmarks, see [../results.md](../results.md). For evaluation workflows, see [evaluate.md](evaluate.md).
