// Serves Ollama's /v1/systemone on top of a vLLM server, so the browser game, the headless runner,
// and the scoring scripts can play a decision model that vLLM serves.
//
//   npm run systemone-vllm -- --vllm http://localhost:8000 --port 11434
//
// Each choice question becomes one /v1/completions request: the prompt Ollama scores
// (src/agent/prompt.ts), one output token restricted to the option letters, and the letters'
// log-probabilities, renormalized over the letters. That is Ollama's softmax over the letter logits,
// the quantity train.py trains. Serve the model under the name clients ask for (--served-model-name)
// and with --logprobs-mode processed_logprobs: by default vLLM returns the top tokens of the whole
// vocabulary, which can leave an option letter out, and the adapter then answers with an error.
import { createServer, type Server } from "node:http";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { entropyConfidence, optionLetters, renderPrompt } from "../src/agent/prompt.ts";

interface ChoiceQuestion {
  type: "choice";
  instructions: string;
  criteria: Record<string, string | null>;
}

interface SystemOneRequest {
  model: string;
  state: unknown;
  questions: Record<string, ChoiceQuestion>;
}

interface Completion {
  choices: { logprobs: { top_logprobs: Record<string, number>[] } }[];
  usage: { prompt_tokens: number };
}

interface Answer {
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
}

class HttpError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

function parseRequest(raw: string): SystemOneRequest {
  let body: Partial<SystemOneRequest>;
  try {
    body = JSON.parse(raw);
  } catch {
    throw new HttpError(400, "request body is not JSON");
  }
  if (typeof body.model !== "string" || !body.questions || typeof body.questions !== "object") {
    throw new HttpError(400, "request needs model and questions");
  }
  for (const [name, q] of Object.entries(body.questions)) {
    if (q?.type !== "choice" || !q.criteria || Object.keys(q.criteria).length < 2) {
      throw new HttpError(400, `question ${name} must be a choice with at least two criteria`);
    }
  }
  return body as SystemOneRequest;
}

export function systemOneVllmServer(vllmBase: string): Server {
  const base = vllmBase.replace(/\/$/, "");
  const letterIds = new Map<string, Promise<number[]>>();

  async function vllm<T>(path: string, body: object): Promise<T> {
    const res = await fetch(base + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const text = await res.text();
    if (!res.ok) {
      let message = text;
      try {
        const parsed = JSON.parse(text);
        message = parsed.error?.message ?? parsed.message ?? text;
      } catch {
        // keep raw text
      }
      throw new HttpError(res.status === 404 ? 404 : 502, `vLLM ${path}: HTTP ${res.status}: ${message}`);
    }
    return JSON.parse(text);
  }

  function idsFor(model: string): Promise<number[]> {
    let ids = letterIds.get(model);
    if (!ids) {
      ids = Promise.all(
        optionLetters(26).map(async (letter) => {
          const { tokens } = await vllm<{ tokens: number[] }>("/tokenize", { model, prompt: letter, add_special_tokens: false });
          if (tokens.length !== 1) throw new HttpError(500, `letter ${letter} is not one token for ${model}`);
          return tokens[0];
        }),
      );
      ids.catch(() => letterIds.delete(model));
      letterIds.set(model, ids);
    }
    return ids;
  }

  async function answer(req: SystemOneRequest, field: string, q: ChoiceQuestion): Promise<{ answer: Answer; inputTokens: number }> {
    const keys = Object.keys(q.criteria);
    const letters = optionLetters(keys.length);
    const ids = (await idsFor(req.model)).slice(0, keys.length);
    const out = await vllm<Completion>("/v1/completions", {
      model: req.model,
      prompt: renderPrompt(req.state, q.instructions, q.criteria, field),
      max_tokens: 1,
      temperature: 0,
      logprobs: keys.length,
      allowed_token_ids: ids,
    });
    const top = out.choices[0].logprobs.top_logprobs[0];
    const missing = letters.filter((l) => !(l in top));
    if (missing.length > 0) {
      throw new HttpError(502, `vLLM left option letters ${missing.join(", ")} out of its log-probabilities; start it with --logprobs-mode processed_logprobs`);
    }
    const mass = letters.map((l) => Math.exp(top[l]));
    const total = mass.reduce((s, v) => s + v, 0);
    const p = mass.map((v) => v / total);
    const best = p.indexOf(Math.max(...p));
    return {
      answer: { choice: keys[best], probabilities: Object.fromEntries(keys.map((k, i) => [k, p[i]])), confidence: entropyConfidence(p) },
      inputTokens: out.usage.prompt_tokens,
    };
  }

  return createServer(async (httpReq, res) => {
    const reply = (status: number, body: object) => {
      res.writeHead(status, { "Content-Type": "application/json" });
      res.end(JSON.stringify(body));
    };
    if (httpReq.method !== "POST" || httpReq.url !== "/v1/systemone") return reply(404, { error: "only POST /v1/systemone is served" });
    let raw = "";
    for await (const chunk of httpReq) raw += chunk;
    try {
      const req = parseRequest(raw);
      const results = await Promise.all(Object.entries(req.questions).map(([field, q]) => answer(req, field, q)));
      const names = Object.keys(req.questions);
      reply(200, {
        model: req.model,
        answers: Object.fromEntries(results.map((r, i) => [names[i], r.answer])),
        usage: { input_tokens: results.reduce((s, r) => s + r.inputTokens, 0) },
      });
    } catch (err) {
      reply(err instanceof HttpError ? err.status : 500, { error: (err as Error).message });
    }
  });
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const { values: args } = parseArgs({
    options: {
      vllm: { type: "string" },
      port: { type: "string", default: "11434" },
      host: { type: "string", default: "127.0.0.1" },
    },
  });
  if (!args.vllm) throw new Error("--vllm <base URL of the vLLM server> is required");
  systemOneVllmServer(args.vllm).listen(Number(args.port), args.host, () =>
    console.log(`/v1/systemone on http://${args.host}:${args.port} -> ${args.vllm}`),
  );
}
