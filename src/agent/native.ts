// Policies for the iOS app (see ios/). The page renders the exact prompt and the
// native side runs it: the fine-tuned decision model is scored from option-letter
// logits, a plain chat model generates JSON under a grammar, and Jev's hosted API
// is called by native code, which holds the key. The bridges are
// WKScriptMessageHandlerWithReply handlers named "decide" and "jev".
import type { EncodedDecision } from "../encoders/types.ts";
import { DecisionApiError, type SystemOneConfig } from "./client.ts";
import { moveGrammar, parseMove, renderChatDecision } from "./llm.ts";
import type { PlayerId } from "./players.ts";
import { systemOnePolicy, type Policy, type PolicyDecision } from "./policies.ts";
import { entropyConfidence, renderDecision } from "./prompt.ts";

interface NativeReply {
  probabilities?: number[];
  tokens?: number;
  ms?: number;
  /** Chat players: the generated JSON answer. */
  text?: string;
  error?: string;
}

interface NativeHandler<T> {
  postMessage(message: unknown): Promise<T>;
}

export type NativeBridge = NativeHandler<NativeReply>;

interface JevReply {
  status?: number;
  body?: string;
  error?: string;
}

function handler<T>(name: string): NativeHandler<T> | null {
  const w = window as unknown as { webkit?: { messageHandlers?: Record<string, NativeHandler<T> | undefined> } };
  return w.webkit?.messageHandlers?.[name] ?? null;
}

export function nativeBridge(): NativeBridge | null {
  return handler<NativeReply>("decide");
}

const MISSING_MODEL_HINT =
  "Copy the model file into the app's Documents folder (see the iPhone section of the README), then pick the player again.";

/** The fine-tuned decision model: one forward pass over the System One prompt, softmax over the option letters. */
export function nativePolicy(bridge: NativeBridge, modelName: string, player: PlayerId = "finetuned"): Policy {
  return {
    name: modelName,
    async decide(enc: EncodedDecision): Promise<PolicyDecision> {
      const started = performance.now();
      const reply = await bridge.postMessage({ player, prompt: renderDecision(enc), options: enc.keys.length });
      if (reply.error || !reply.probabilities) {
        throw new DecisionApiError("model-missing", reply.error ?? "no reply from the on-device model", MISSING_MODEL_HINT);
      }
      const p = reply.probabilities;
      const probabilities: Record<string, number> = {};
      enc.keys.forEach((k, i) => (probabilities[k] = p[i]));
      let best = 0;
      p.forEach((v, i) => (best = v > p[best] ? i : best));
      return {
        choice: enc.keys[best],
        probabilities,
        confidence: entropyConfidence(p),
        inputTokens: reply.tokens,
        latencyMs: reply.ms ?? performance.now() - started,
      };
    },
  };
}

/**
 * A plain chat model on device, asked the way the `llm:` policy asks Ollama: the teacher prompt with the
 * JSON answer line, in the model's chat template, generated at temperature 0 under a grammar that only
 * admits {"move": "<option>"}. Probabilities come from the logits at the move token.
 */
export function nativeChatPolicy(bridge: NativeBridge, modelName: string, player: PlayerId = "phi4-mini"): Policy {
  return {
    name: modelName,
    async decide(enc: EncodedDecision): Promise<PolicyDecision> {
      const started = performance.now();
      const reply = await bridge.postMessage({ player, prompt: renderChatDecision(enc), grammar: moveGrammar(enc.keys), keys: enc.keys });
      if (reply.error || reply.text === undefined) {
        throw new DecisionApiError("model-missing", reply.error ?? "no reply from the on-device model", MISSING_MODEL_HINT);
      }
      const choice = parseMove(reply.text, enc.keys);
      let probabilities: Record<string, number> | undefined;
      if (reply.probabilities?.length === enc.keys.length) {
        probabilities = {};
        for (const [i, k] of enc.keys.entries()) probabilities[k] = reply.probabilities[i];
      }
      return {
        choice,
        probabilities,
        confidence: probabilities?.[choice],
        inputTokens: reply.tokens,
        latencyMs: reply.ms ?? performance.now() - started,
      };
    },
  };
}

/**
 * A fetch that sends a request body through the native "jev" handler. The native side posts it to Jev's
 * fixed endpoint with the key it reads from the app's Documents, so the key never reaches the page.
 */
export function nativeJevFetch(bridge: NativeHandler<JevReply>): typeof fetch {
  return async (_input, init) => {
    const reply = await bridge.postMessage({ body: String(init?.body ?? "") });
    if (init?.signal?.aborted) throw new DOMException("aborted", "AbortError");
    if (reply.error || reply.status === undefined) {
      throw new DecisionApiError(
        "network",
        reply.error ?? "no reply from the native Jev bridge",
        "Jev needs a network connection and the API key in the app's Documents as jev_key.txt (see the iPhone section of the README).",
      );
    }
    return new Response(reply.body ?? "", { status: reply.status });
  };
}

/** TypeSafe's hosted Jev, reached through the native bridge. Null outside the app. */
export function nativeJevPolicy(cfg: Pick<SystemOneConfig, "model"> & { baseUrl?: string }): Policy | null {
  const bridge = handler<JevReply>("jev");
  if (!bridge) return null;
  return systemOnePolicy({ baseUrl: cfg.baseUrl ?? "https://api.typesafe.ai", model: cfg.model, hosted: true, fetch: nativeJevFetch(bridge) });
}
