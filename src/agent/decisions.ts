// OpenAI's Decisions API (POST /v1/decisions, model gpt-6-luna) as a player. It is a decision model
// in the same sense as Jev: it reads shared evidence (`input`) and answers a `choice` question with a
// probability for every option, instead of generating text. The request carries the same facts Jev gets:
// the encoded state as the input, the encoder's instructions as the question, and the legal options as
// the choices (with the encoder's descriptions, e.g. "turn around now" for back).
import type { EncodedDecision } from "../encoders/types.ts";
import type { Policy, PolicyDecision } from "./policies.ts";

export const DECISIONS_DEFAULT_BASE_URL = "https://api.openai.com";

export interface DecisionsConfig {
  /** Base URL without /v1, for example https://api.openai.com, or /openai behind the dev server's proxy. */
  baseUrl: string;
  model: string;
  /** Bearer key. Leave unset when a proxy adds it (the browser dev server). Never logged. */
  apiKey?: string;
  timeoutMs?: number;
  /** Request function, replaceable in tests. */
  fetchFn?: typeof fetch;
}

/** The model refused, answered outside the legal options, or the reply had no usable answer. */
export class DecisionsAnswerError extends Error {}

export type ChoiceInput = Pick<EncodedDecision, "state" | "instructions" | "criteria"> & { keys: string[] };

export function buildDecisionsRequest(model: string, enc: ChoiceInput): object {
  const state = typeof enc.state === "string" ? enc.state : JSON.stringify(enc.state);
  return {
    model,
    input: state,
    questions: [
      {
        type: "choice",
        name: "move",
        instructions: enc.instructions,
        choices: enc.keys.map((k) => (enc.criteria[k] ? { value: k, description: enc.criteria[k] } : { value: k })),
      },
    ],
  };
}

interface DecisionsAnswer {
  type: string;
  name?: string;
  choice?: string;
  probabilities?: { value: string; probability: number }[];
  confidence?: number;
}

export interface DecisionsBody {
  answers?: DecisionsAnswer[];
  usage?: { input_tokens?: number };
}

/** Reads the `move` answer. Throws a DecisionsAnswerError on a refusal or a choice outside the legal options. */
export function parseDecisionsAnswer(
  body: DecisionsBody,
  keys: string[],
): { choice: string; probabilities: Record<string, number>; confidence?: number; inputTokens?: number } {
  const answer = body.answers?.find((a) => a.name === "move") ?? body.answers?.[0];
  if (!answer) throw new DecisionsAnswerError("no answer in reply");
  if (answer.type === "refusal") throw new DecisionsAnswerError("refusal");
  if (answer.type !== "choice") throw new DecisionsAnswerError(`unexpected answer type "${answer.type}"`);
  const choice = String(answer.choice ?? "");
  if (!keys.includes(choice)) throw new DecisionsAnswerError(`illegal move "${choice}" (options: ${keys.join(", ")})`);
  const probabilities: Record<string, number> = {};
  for (const p of answer.probabilities ?? []) if (keys.includes(p.value)) probabilities[p.value] = p.probability;
  return { choice, probabilities, confidence: answer.confidence, inputTokens: body.usage?.input_tokens };
}

export function decisionsPolicy(cfg: DecisionsConfig): Policy {
  return {
    name: `openai:${cfg.model}`,
    async decide(enc, signal): Promise<PolicyDecision> {
      const started = performance.now();
      const res = await (cfg.fetchFn ?? fetch)(`${cfg.baseUrl.replace(/\/$/, "")}/v1/decisions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(cfg.apiKey ? { Authorization: `Bearer ${cfg.apiKey}` } : {}) },
        body: JSON.stringify(buildDecisionsRequest(cfg.model, enc)),
        signal: signal ?? AbortSignal.timeout(cfg.timeoutMs ?? 10_000),
      });
      const text = await res.text();
      const latencyMs = performance.now() - started;
      if (!res.ok) throw new Error(`decisions HTTP ${res.status}: ${text.slice(0, 300)}`);
      return { ...parseDecisionsAnswer(JSON.parse(text) as DecisionsBody, enc.keys), latencyMs };
    },
  };
}
