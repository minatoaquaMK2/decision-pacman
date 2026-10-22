// A plain chat model as a player: the teacher's prompt over the same encoded
// state, sent to Ollama's /api/chat with a JSON schema that only admits the
// legal options. Thinking off, temperature 0. Option probabilities come from
// the token logprobs at the position where the move is written.
import type { EncodedDecision } from "../encoders/types.ts";
import type { Policy, PolicyDecision } from "./policies.ts";
import { teacherPrompt } from "./teacher.ts";

export interface LlmConfig {
  /** Ollama base URL without /api, for example http://localhost:11434. */
  baseUrl: string;
  model: string;
  keepAlive?: string | number;
  timeoutMs?: number;
}

export const LLM_ANSWER = 'Answer with only a JSON object: {"move": "<one of the options>"}';

export class LlmAnswerError extends Error {}

/** JSON schema for the reply: one field, `move`, restricted to the legal options. */
export function moveSchema(keys: string[]): object {
  return {
    type: "object",
    properties: { move: { type: "string", enum: keys } },
    required: ["move"],
  };
}

/**
 * The GBNF grammar equivalent of `moveSchema` for llama.cpp's grammar sampler, in the shape llama.cpp's
 * json-schema-to-grammar gives for it: optional whitespace around the tokens, the move as a JSON string
 * from the options. Used on device, where there is no Ollama to translate the schema.
 */
export function moveGrammar(keys: string[]): string {
  const values = keys.map((k) => `"\\"${k}\\""`).join(" | ");
  return [
    'root ::= "{" space move-kv "}" space',
    'move-kv ::= "\\"move\\"" space ":" space move',
    `move ::= (${values}) space`,
    'space ::= | " " | "\\n"{1,2} [ \\t]{0,20}',
  ].join("\n");
}

/** One user message in phi4-mini's chat template, exactly as Ollama renders it (no system prompt), ending where the reply starts. */
export function phi4MiniPrompt(user: string): string {
  return `<|user|>${user}<|end|><|assistant|>`;
}

/** The full on-device prompt for a plain chat model: the same text `buildChatRequest` sends, in phi4-mini's template. */
export function renderChatDecision(enc: EncodedDecision): string {
  return phi4MiniPrompt(teacherPrompt(enc, undefined, LLM_ANSWER));
}

export function buildChatRequest(cfg: LlmConfig, enc: EncodedDecision): object {
  return {
    model: cfg.model,
    messages: [{ role: "user", content: teacherPrompt(enc, undefined, LLM_ANSWER) }],
    format: moveSchema(enc.keys),
    think: false,
    stream: false,
    keep_alive: cfg.keepAlive ?? -1,
    logprobs: true,
    top_logprobs: 10,
    options: { temperature: 0, num_predict: 32 },
  };
}

export interface TokenLogprob {
  token: string;
  logprob: number;
  top_logprobs?: { token: string; logprob: number }[];
}

/** Reads the move from the reply. Throws unless it is one of the options. */
export function parseMove(content: string, keys: string[]): string {
  let obj: { move?: unknown };
  try {
    obj = JSON.parse(content);
  } catch {
    throw new LlmAnswerError(`unparseable answer: ${content.slice(0, 200)}`);
  }
  const move = String(obj?.move ?? "").trim().toLowerCase();
  if (!keys.includes(move)) throw new LlmAnswerError(`illegal move "${move}" (options: ${keys.join(", ")})`);
  return move;
}

/**
 * Option probabilities from the logprobs of the token that starts the move value.
 * Candidate tokens count toward an option when the option's name starts with them;
 * the result is renormalized over the options. Returns undefined when the runtime
 * gave no logprobs or none of the candidates is an option.
 */
export function moveProbabilities(content: string, logprobs: TokenLogprob[] | undefined, keys: string[]): Record<string, number> | undefined {
  if (!logprobs?.length) return undefined;
  const colon = content.indexOf(":", content.indexOf('"move"'));
  const valueStart = content.indexOf('"', colon) + 1;
  if (colon < 0 || valueStart <= 0) return undefined;
  let end = 0;
  const at = logprobs.find((t) => (end += t.token.length) > valueStart);
  if (!at) return undefined;
  const mass: Record<string, number> = Object.fromEntries(keys.map((k) => [k, 0]));
  for (const cand of at.top_logprobs ?? [at]) {
    const text = cand.token.replace(/^[\s"]+/, "").toLowerCase();
    if (!text) continue;
    for (const k of keys) if (k.startsWith(text)) mass[k] += Math.exp(cand.logprob);
  }
  const total = keys.reduce((a, k) => a + mass[k], 0);
  if (total <= 0) return undefined;
  return Object.fromEntries(keys.map((k) => [k, mass[k] / total]));
}

export function llmPolicy(cfg: LlmConfig): Policy {
  return {
    name: `llm:${cfg.model}`,
    async decide(enc, signal): Promise<PolicyDecision> {
      const started = performance.now();
      const res = await fetch(`${cfg.baseUrl.replace(/\/$/, "")}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildChatRequest(cfg, enc)),
        signal: signal ?? AbortSignal.timeout(cfg.timeoutMs ?? 30_000),
      });
      if (!res.ok) throw new Error(`llm HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
      const body = (await res.json()) as {
        message?: { content?: string };
        logprobs?: TokenLogprob[];
        prompt_eval_count?: number;
      };
      const content = body.message?.content ?? "";
      const choice = parseMove(content, enc.keys);
      const probabilities = moveProbabilities(content, body.logprobs, enc.keys);
      return {
        choice,
        probabilities,
        confidence: probabilities?.[choice],
        latencyMs: performance.now() - started,
        inputTokens: body.prompt_eval_count,
      };
    },
  };
}
