// A chat model used as a teacher: it sees the same encoded state as the
// student and answers with one option plus a short reason.
import type { EncodedDecision } from "../encoders/types.ts";
import { TPS } from "../engine/game.ts";
import { peek, type PeekOutcome } from "./oracle.ts";
import type { Policy, PolicyDecision } from "./policies.ts";

export interface TeacherConfig {
  /** OpenAI-compatible base URL, ending in /v1. */
  baseUrl: string;
  model: string;
  /** Sent as `Authorization: Bearer <apiKey>` when set (hosted APIs). Local servers usually need none. */
  apiKey?: string;
  /**
   * Fields merged into every request body, overriding the defaults. Hosted APIs ignore `chat_template_kwargs`,
   * so turning thinking off there needs their own switch, e.g. `{"reasoning": {"enabled": false}}` on OpenRouter
   * and Vercel AI Gateway.
   */
  extraBody?: Record<string, unknown>;
  /** Let the model think before answering (slower, usually more accurate). */
  think?: boolean;
  /** Show the model one simulated future per option, this many seconds long. Needs the live game state. */
  peekSeconds?: number;
  timeoutMs?: number;
  /** Retries after HTTP 429 or 5xx, honoring Retry-After. Default 0 (fail at once, as before). */
  maxRetries?: number;
  /** First backoff when the server gives no Retry-After; doubles per retry, capped at 60 s. Default 2000. */
  retryBaseMs?: number;
  /**
   * Answers by exact prompt. A repeated prompt is answered from here without a request; the answer is deterministic
   * at temperature 0, and lockstep play revisits identical states often.
   */
  cache?: Map<string, TeacherDecision>;
  /** Request function, replaceable in tests. */
  fetchFn?: typeof fetch;
}

export interface TeacherDecision extends PolicyDecision {
  reason: string;
  outputTokens: number;
  /** Requests repeated after a 429 or 5xx before this answer. */
  retries?: number;
  /** Hidden reasoning tokens, when the server reports them. Nonzero means thinking was not off. */
  reasoningTokens?: number;
  /** Cost in USD, when the server reports it (OpenRouter and Vercel AI Gateway do, in `usage.cost`). */
  costUsd?: number;
  /** True when the answer came from `cache` instead of the server. */
  cached?: boolean;
}

export class TeacherParseError extends Error {}

export const TEACHER_ANSWER = 'Answer with only a JSON object: {"move": "<one of the options>", "reason": "<one short sentence>"}';

/** The rules, the encoded state, and the options, followed by `answer`, the line that says how to reply. */
export function teacherPrompt(
  enc: EncodedDecision,
  peeks?: { seconds: number; outcomes: Record<string, PeekOutcome> },
  answer: string = TEACHER_ANSWER,
): string {
  const state = typeof enc.state === "string" ? enc.state : JSON.stringify(enc.state);
  const options = enc.keys.map((k) => `- ${k}${enc.criteria[k] ? `: ${enc.criteria[k]}` : ""}`).join("\n");
  const lookahead = peeks
    ? `Lookahead: for each option, the game engine played it and then simple greedy moves for the next ${peeks.seconds} s. ` +
      "Normal ghost moves are exact; frightened ghosts turn at random. It shows one possible future, not the only one.\n" +
      `${JSON.stringify(peeks.outcomes)}\n\n`
    : "";
  return `${enc.instructions}\n\nState:\n${state}\n\nOptions:\n${options}\n\n${lookahead}${answer}`;
}

/** Extracts {"move", "reason"} from the reply. Throws unless the move is one of the options. */
export function parseTeacherReply(text: string, keys: string[]): { move: string; reason: string } {
  const matches = text.match(/\{[^{}]*"move"[^{}]*\}/g);
  if (!matches) throw new TeacherParseError(`no JSON answer in reply: ${text.slice(-200)}`);
  const last = matches[matches.length - 1];
  let obj: { move?: unknown; reason?: unknown };
  try {
    obj = JSON.parse(last);
  } catch {
    throw new TeacherParseError(`unparseable answer: ${last}`);
  }
  const move = String(obj.move ?? "").trim().toLowerCase();
  if (!keys.includes(move)) throw new TeacherParseError(`illegal move "${move}" (options: ${keys.join(", ")})`);
  return { move, reason: String(obj.reason ?? "") };
}

export function teacherPolicy(cfg: TeacherConfig): Policy {
  return {
    name: `teacher:${cfg.model}${cfg.think ? ":think" : ""}${cfg.peekSeconds ? `:peek${cfg.peekSeconds}s` : ""}`,
    async decide(enc, _signal, game): Promise<TeacherDecision> {
      const started = performance.now();
      let peeks: { seconds: number; outcomes: Record<string, PeekOutcome> } | undefined;
      if (cfg.peekSeconds) {
        if (!game) throw new Error("peeking needs the live game state");
        const outcomes: Record<string, PeekOutcome> = {};
        for (const key of enc.keys) outcomes[key] = peek(game, enc, key, Math.round(cfg.peekSeconds * TPS));
        peeks = { seconds: cfg.peekSeconds, outcomes };
      }
      const prompt = teacherPrompt(enc, peeks);
      const hit = cfg.cache?.get(prompt);
      if (hit) return { ...hit, latencyMs: performance.now() - started, cached: true };
      const send = () =>
        (cfg.fetchFn ?? fetch)(`${cfg.baseUrl.replace(/\/$/, "")}/chat/completions`, {
          method: "POST",
          headers: { "Content-Type": "application/json", ...(cfg.apiKey ? { Authorization: `Bearer ${cfg.apiKey}` } : {}) },
          body: JSON.stringify({
            model: cfg.model,
            messages: [{ role: "user", content: prompt }],
            temperature: 0,
            max_tokens: cfg.think ? 4096 : 200,
            chat_template_kwargs: { enable_thinking: !!cfg.think },
            ...cfg.extraBody,
          }),
          signal: AbortSignal.timeout(cfg.timeoutMs ?? 120_000),
        });
      // Servers close keep-alive connections left idle past their timeout, and a request sent on one as it closes
      // fails before any response (fetch throws a TypeError). One more attempt goes out on a new connection.
      const request = () => send().catch((err: unknown) => (err instanceof TypeError ? send() : Promise.reject(err)));
      // Hosted APIs rate-limit (429) and occasionally fail (5xx): wait and retry instead of losing the state.
      let res = await request();
      let attemptStarted = started;
      let retries = 0;
      for (; !res.ok && retryable(res.status) && retries < (cfg.maxRetries ?? 0); retries++) {
        await sleep(await retryDelayMs(res, retries, cfg.retryBaseMs ?? 2000));
        attemptStarted = performance.now();
        res = await request();
      }
      if (!res.ok) throw new Error(`teacher HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
      const body = (await res.json()) as {
        choices: { message: { content?: string } }[];
        usage?: {
          prompt_tokens?: number;
          completion_tokens?: number;
          completion_tokens_details?: { reasoning_tokens?: number };
          cost?: number;
        };
      };
      const text = body.choices[0]?.message?.content ?? "";
      const { move, reason } = parseTeacherReply(text, enc.keys);
      const decision: TeacherDecision = {
        choice: move,
        reason,
        // The successful request's own latency; time spent waiting out rate limits is not included.
        latencyMs: performance.now() - attemptStarted,
        ...(retries ? { retries } : {}),
        inputTokens: body.usage?.prompt_tokens,
        outputTokens: body.usage?.completion_tokens ?? 0,
        reasoningTokens: body.usage?.completion_tokens_details?.reasoning_tokens,
        costUsd: body.usage?.cost,
      };
      cfg.cache?.set(prompt, decision);
      return decision;
    },
  };
}

function retryable(status: number): boolean {
  return status === 429 || status >= 500;
}

/** Retry-After (seconds) if present, else "retry after Ns" in the body, else exponential backoff. */
async function retryDelayMs(res: Response, attempt: number, baseMs: number): Promise<number> {
  const header = Number(res.headers.get("retry-after"));
  if (Number.isFinite(header) && header > 0) return header * 1000 + 250;
  const text = await res.text().catch(() => "");
  const m = /retry after (\d+(?:\.\d+)?)\s*s/i.exec(text);
  if (m) return Number(m[1]) * 1000 + 250;
  return Math.min(60_000, baseMs * 2 ** attempt);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
