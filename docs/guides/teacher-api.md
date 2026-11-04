# Label with a hosted teacher

The published labels came from Qwen3.8-27B (NVFP4 weights) on one RTX 5090. Without that GPU, a hosted API can be the teacher. This guide covers what to set so `gen-data` talks to a hosted API, a cheap option that was measured (DeepSeek V4.1 Flash on Ollama Cloud), who serves the published teacher model itself, and what relabeling the published run's 54,325 requests costs.

**Short version.** DeepSeek V4.1 Flash on Ollama Cloud labeled 22 real states with no failures, 455 input and 31 output tokens per request, and 0.6 s per request. At Ollama's list prices, 54,325 requests would cost about $4.73 off-peak or $9.46 at peak. It is not the model the published labels came from: it chose the published teacher's move on 21 of those 22 states, a sample far too small to call it equivalent. To reproduce the published labels as closely as possible, use a provider that serves Qwen3.8-27B (from about $6.90 for the same run, estimated with the same token counts).

Labeling is the only place a hosted model enters. The student trains and plays locally as in [distill.md](distill.md).

## What differs from a local server

Four things, all handled through `.env`:

1. **Key.** `TEACHER_API_KEY` is sent as `Authorization: Bearer <key>`. Keep it out of the repository; `.env` is gitignored.
2. **Thinking.** Qwen3.8 and DeepSeek V4.1 Flash think by default. The teacher asks for thinking off with `chat_template_kwargs: {"enable_thinking": false}`, which a local vLLM or SGLang server honors and hosted APIs ignore. With thinking left on, a hosted model spends its 200-token answer budget on hidden reasoning (on a two-word test prompt it used 105 reasoning tokens for a two-token answer), so answers come back truncated or not at all and cost several times more. Each provider has its own switch; put it in `TEACHER_EXTRA_BODY`, a JSON object merged into every request (see the table below). Check that the report's `reasoningTokens` is 0 after a short run.
3. **Rate limits.** Hosted APIs answer HTTP 429 when you exceed a limit. With `TEACHER_MAX_RETRIES` above 0, the teacher waits (honoring `Retry-After`, or "retry after N s" in the error text) and asks again instead of dropping the state. The default 0 keeps the old behavior: a failed request leaves the state unlabeled.
4. **Concurrency.** `gen-data` plays in lockstep, so one process has one request in flight. Concurrency is the number of processes you run. Run one process per seed range, or per seed, to fill the provider's limit.

`gen-data` now records each request's `label_input_tokens`, `label_output_tokens`, and, when the server reports it (OpenRouter and Vercel AI Gateway do, in `usage.cost`), `label_cost_usd`; the report sums them.

## Recommended cheap teacher: DeepSeek V4.1 Flash on Ollama Cloud

Ollama Cloud serves `deepseek-v4.1-flash` (a 763B-parameter mixture-of-experts model, FP8, per Ollama's model metadata) behind an OpenAI-compatible endpoint. Billing is pay-as-you-go from usage credits: the Free plan includes starter credits, and Pro ($20 a month) and Max ($100 a month) include $60 and $300 of credits. List prices on 2026-10-05, per million tokens:

| | Input | Cached input | Output |
|---|---|---|---|
| Peak (12:00-18:00 UTC on weekdays) | $0.30 | $0.006 | $1.20 |
| Off-peak (all other hours, and weekends) | $0.15 | $0.003 | $0.60 |

Max lists 10 concurrent requests; the pricing page states no concurrency number for Free or Pro. Sources: [Ollama pricing](https://ollama.com/pricing), the [model page](https://ollama.com/library/deepseek-v4.1-flash), and Ollama's [cloud](https://docs.ollama.com/cloud) and [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility) docs.

```bash
TEACHER_BASE_URL=https://ollama.com/v1
TEACHER_MODEL=deepseek-v4.1-flash
TEACHER_API_KEY=<your ollama.com API key>
TEACHER_EXTRA_BODY={"reasoning_effort":"none"}
TEACHER_MAX_RETRIES=5
```

`{"reasoning_effort":"none"}` (or `{"reasoning":{"effort":"none"}}`) turns thinking off on this endpoint; `{"think":false}` did not. Ollama does not return a cost in `usage`, so `label_cost_usd` stays empty; compute cost from the recorded tokens and the price for the hour you ran in.

Measured numbers are under [Measured](#measured). The caveat stands throughout: a different teacher gives different labels, and a student trained on them is a new model to evaluate, not a reproduction of `pacman-0.8b-qwen`.

## Providers of Qwen3.8-27B

Serverless endpoints for Qwen3.8-27B, the model behind the published labels, as listed on 2026-10-05. Prices are USD per million tokens. "Per 1k requests" applies the token counts measured on DeepSeek V4.1 Flash with the same prompt, 455.4 input and 31.3 output tokens per request (see [Measured](#measured)): (455.4 × input price + 31.3 × output price) / 1,000. Qwen's tokenizer counts the same prompt somewhat differently, so treat these as estimates and measure a small batch on your provider before a full run.

| Provider | Reached through | Input | Output | Per 1k requests | Weights | Notes |
|---|---|---|---|---|---|---|
| DeepInfra | direct, OpenRouter, Vercel AI Gateway | 0.15 | 1.875 | $0.127 | bf16 | 200 concurrent requests per model by default; structured output; no logprobs listed on OpenRouter |
| Parasail | OpenRouter, Vercel AI Gateway | 0.24 | 2.20 | $0.178 | fp8 | logprobs and structured output |
| Chutes | OpenRouter | 0.24 | 2.20 | $0.178 | fp8 | |
| Alibaba Cloud Model Studio | direct | 0.424 (China, Beijing) / 0.50 (International) | 1.696 / 3.00 | $0.246 / $0.322 | official | the model's publisher; TPM limit tiered by monthly spend; logprobs |
| Alibaba | OpenRouter / Vercel AI Gateway | 0.425 / 0.50 | 2.55 / 3.00 | $0.273 / $0.322 | official | |
| Novita | OpenRouter, Vercel AI Gateway | 0.42 | 3.00 | $0.285 | not stated | logprobs |
| Groq | direct | 0.80 | 4.00 | $0.490 | Groq TruePoint | preview model; developer plan 250K tokens and 1K requests per minute; 131K context |
| Cerebras | OpenRouter, Vercel AI Gateway | 0.99 | 1.49 | $0.497 | fp16 | logprobs; 65K context on OpenRouter |

OpenRouter also lists smaller hosts with lower input prices: Darkbloom (fp4, 0.05 / 2.20, $0.101 per 1k requests), Ionstream (fp8, 0.089 / 2.35, $0.124), DekaLLM (0.049 / 3.00), Reka (0.05 / 3.00), and Wafer (0.024 / 4.35, $0.164). Their quantization and capacity are less documented. Check uptime on OpenRouter before you start a long run with one of them.

Not usable for this job on 2026-10-05: Together AI offers Qwen3.8-27B only for fine-tuning and dedicated inference, and Fireworks only as an on-demand deployment ("Serverless: Not supported"). OpenRouter's free variant `qwen/qwen3.8-27b:free` is capped at 50 requests a day (1,000 after buying $10 of credits). Vercel AI Gateway's free tier refuses Qwen3.8-27B ("Free tier users do not have access to this model").

Sources, read 2026-10-05: OpenRouter's endpoint list (`https://openrouter.ai/api/v1/models/qwen/qwen3.8-27b/endpoints`), Vercel AI Gateway's (`https://ai-gateway.vercel.sh/v1/models/alibaba/qwen3.8-27b/endpoints`), the model pages of [Alibaba Cloud Model Studio](https://www.alibabacloud.com/help/en/model-studio/qwen3-8-27b), [Groq](https://console.groq.com/docs/model/qwen/qwen3.8-27b), and [Fireworks](https://fireworks.ai/models/fireworks/qwen3p8-27b), [DeepInfra's rate limits](https://docs.deepinfra.com/account/rate-limits), and [OpenRouter's limits](https://openrouter.ai/docs/api_reference/limits). Prices change; recheck them before a run.

## Settings

Set the teacher in `.env` (gitignored). Take the key from your password manager or the provider's console; never commit it.

| Route | `TEACHER_BASE_URL` | `TEACHER_MODEL` | `TEACHER_EXTRA_BODY` (thinking off) |
|---|---|---|---|
| OpenRouter | `https://openrouter.ai/api/v1` | `qwen/qwen3.8-27b` | `{"reasoning":{"effort":"none"}}` |
| Vercel AI Gateway | `https://ai-gateway.vercel.sh/v1` | `alibaba/qwen3.8-27b` | `{"reasoning":{"enabled":false}}` |
| Alibaba Cloud Model Studio | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `qwen3.8-27b` | `{"enable_thinking":false}` |
| Groq | `https://api.groq.com/openai/v1` | `qwen/qwen3.8-27b` | `{"reasoning_effort":"none"}` |
| Ollama Cloud (DeepSeek V4.1 Flash, not Qwen) | `https://ollama.com/v1` | `deepseek-v4.1-flash` | `{"reasoning_effort":"none"}` |

The Ollama Cloud switch was run here and gave 0 reasoning tokens. The other switches are from each provider's documentation. Whichever route you use, check that the recorded reasoning tokens are 0 on a few requests before a long run; `chat_template_kwargs` alone is not enough on hosted APIs.

An aggregator routes each request to any provider that serves the model unless you pin one. Pin the provider you priced, so the labels come from one set of weights, by adding to the same JSON:

- OpenRouter: `"provider":{"order":["deepinfra"],"allow_fallbacks":false}`
- Vercel AI Gateway: `"providerOptions":{"gateway":{"only":["deepinfra"]}}`

For example, DeepInfra through Vercel AI Gateway:

```bash
TEACHER_BASE_URL=https://ai-gateway.vercel.sh/v1
TEACHER_MODEL=alibaba/qwen3.8-27b
TEACHER_API_KEY=<your gateway key>
TEACHER_EXTRA_BODY={"reasoning":{"enabled":false},"providerOptions":{"gateway":{"only":["deepinfra"]}}}
TEACHER_MAX_RETRIES=30
```

Then run the commands of [distill.md](distill.md#1-label-states) unchanged. Because each process has one request in flight, start more processes than the published run's eight to use a hosted provider's capacity, one per seed or per small seed range:

```bash
for seed in $(seq 2000 2029); do
  npm run gen-data -- --player greedy --labeler teacher-peek5s --games 1 --seed $seed --out data/q1/train > data/q1/logs/g$seed.log 2>&1 &
done
wait
```

The answer cache is per process, so splitting a seed range across processes sends a few more requests than the published run's 54,325 (repeated prompts within one game are still cached).

## Measure before you relabel

Try a provider on a sample before paying for the full set. `--label-every k` asks the teacher on every k-th decision only, while the player still plays every one, so the sample covers whole games:

```bash
npm run gen-data -- --player greedy --labeler teacher-peek5s --games 2 --seed 2000 --label-every 20 --out data/api/train
```

Then compare with labels you already have for the same players and seeds, and project the cost:

```bash
python3 scripts/compare_teacher_labels.py --new data/api/train --ref data/q1/train \
  --price deepinfra:0.15:1.875 --price alibaba-intl:0.50:3.00 --requests 54325
```

Players are deterministic for a seed, so the script joins the two runs on player, seed, and tick and checks that the encoded state and options are identical before comparing labels. It reports agreement (with a 95% interval), tokens, latency, errors, the cost the server reported, and the projection.

## Measured

One small measurement on 2026-10-05 (Pacific time), with the `teacher-peek5s` prompt of the published run (features encoding plus the 5-second rollout facts), temperature 0, thinking off, on training seeds. The model is not the published teacher, and the sample is small: it establishes tokens, cost, latency, and that the setup works, not label quality.

### DeepSeek V4.1 Flash, Ollama Cloud

One process (concurrency 1), `--player greedy --seed 2000 --games 1 --label-every 60`, so the 22 states are spread over the whole game. Run at 00:28 UTC on 2026-10-06, off-peak.

| | Value |
|---|---|
| Requests answered | 22 of 22 |
| Input tokens per request | 455.4 mean, 514 p90 |
| Output tokens per request | 31.3 mean, 35 p90 |
| Reasoning tokens | 0 |
| Latency per request | 0.60 s p50, 0.78 s p90 |
| Unparseable or illegal answers, HTTP errors | 0 |
| Cost of the run | 10,019 input and 688 output tokens: (10,019 × $0.15 + 688 × $0.60) / 10^6 = $0.0019 off-peak |
| Same choice as the published Qwen3.8-27B labels | 21 of 22 (95% interval 78-99%); tiny sample, a rough signal only |

### Projected for the published run

The published run sent 54,325 teacher requests. Cost = 54,325 × (input tokens × input price + output tokens × output price) / 10^6, with the 455.4 input and 31.3 output tokens per request measured on DeepSeek V4.1 Flash; the Qwen3.8-27B rows reuse those counts as an estimate. Cached-input discounts are ignored, so these are upper bounds where a provider caches the shared prompt prefix.

| Provider (Qwen3.8-27B unless noted) | Per 1k requests | 54,325 requests |
|---|---|---|
| DeepSeek V4.1 Flash, Ollama Cloud, off-peak | 0.0683 + 0.0188 = $0.087 | $4.73 |
| DeepSeek V4.1 Flash, Ollama Cloud, peak | 0.1366 + 0.0376 = $0.174 | $9.46 |
| DeepInfra | 0.0683 + 0.0587 = $0.127 | $6.90 |
| Parasail | 0.1093 + 0.0689 = $0.178 | $9.68 |
| Alibaba Model Studio, China (Beijing) | 0.1931 + 0.0531 = $0.246 | $13.37 |
| Alibaba Model Studio, International (also via Vercel AI Gateway) | 0.2277 + 0.0939 = $0.322 | $17.47 |
| Groq | 0.3643 + 0.1252 = $0.490 | $26.59 |
| Cerebras | 0.4508 + 0.0466 = $0.497 | $27.03 |

Wall time depends on latency and on how many processes run, not on price. With P processes and latency L per request, the run takes about 54,325 × L / P, but never less than the longest single game: up to 1,276 requests in one game of the published run, asked one after another. Vercel AI Gateway's own latency figures for Qwen3.8-27B over the preceding hour were about 0.6-0.7 s p50 for DeepInfra, Parasail, Cerebras, and Novita, and 2.1 s for Alibaba. Except for the DeepSeek latency measured here, these are estimates:

| | L | P = 16 | P = 32 | One game, floor |
|---|---|---|---|---|
| DeepSeek V4.1 Flash, Ollama Cloud (measured) | 0.6 s | 34 min | 17 min | 13 min |
| DeepInfra-class latency | 0.7 s | 40 min | 20 min | 15 min |
| Alibaba-class latency | 2.1 s | 119 min | 59 min | 45 min |

On Ollama Cloud, Max's 10 concurrent requests give 54,325 × 0.6 s / 10 = 54 minutes; at 2 concurrent requests the run takes about 4.5 hours. Groq's developer plan caps tokens at 250K per minute: at about 502 tokens per request that is about 500 requests per minute, so at least 109 minutes. The published run on the RTX 5090 took about 65 minutes at concurrency 8.

## Caveats

- The hosted teacher is not the published teacher. Same model name is not same numbers: the published labels came from NVFP4 weights on a 5090, and providers serve bf16, fp8, fp4, or unstated precision. Relabel a sample and compare (above) before trusting a provider's labels for a full run, and pin the provider.
- DeepSeek V4.1 Flash on Ollama Cloud is not the published teacher, and the 22-state check above only confirms the prompt and answer format. In section 23 of [model_evaluation.md](../model_evaluation.md), relabeling all 55,588 requests with DeepSeek-V4-Flash on an in-cluster vLLM server agreed with Qwen on 91.1% of training states and trained a student that matched Qwen's student in pellets and survival while eating about 3 fewer frightened ghosts per game.
- Ollama Cloud prices depend on the hour (peak 12:00-18:00 UTC on weekdays costs twice off-peak), and its pages do not state concurrency limits for Free and Pro. Run a short pilot at the concurrency you plan to use.
- Free tiers are not enough for a full run: OpenRouter's free variant allows 50 to 1,000 requests a day, and Vercel AI Gateway's free tier refuses Qwen3.8-27B.
- Hosted answers at temperature 0 are not guaranteed to be deterministic across providers or over time, so a rerun can differ in a few labels.
- Logprobs are listed on OpenRouter for Alibaba, Parasail, Novita, and Cerebras, not for DeepInfra. Set `TEACHER_LOGPROBS=1` to make `gen-data` request token logprobs and record `label_probs` for `training/build_sft.py --target probs`. The published run did not request logprobs, and in section 24 of [model_evaluation.md](../model_evaluation.md) soft teacher probabilities performed the same as smoothed hard labels.
