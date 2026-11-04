import { createHash } from "node:crypto";
import { appendFileSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { cpus, platform, arch } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { buildDecisionsRequest, DecisionsAnswerError, parseDecisionsAnswer, type ChoiceInput, type DecisionsBody } from "../src/agent/decisions.ts";
import type { PolicyDecision } from "../src/agent/policies.ts";
import { ENCODERS } from "../src/encoders/index.ts";
import { percentile, playGame, type GameSummary } from "../src/sim/runner.ts";

export const PRICING = {
  model: "gpt-6-luna",
  checkedDate: "2026-11-04",
  usdPerMillionTokens: { uncachedInput: 0.10, cachedInput: 0.01, cacheWrite: 0.125, output: 0 },
  sources: ["https://developers.openai.com/api/docs/guides/decisions", "https://developers.openai.com/api/docs/models/gpt-6-luna"],
  basis: "API response usage at public list prices; not an invoice",
};

const JEV_PRICING = {
  checkedDate: "2026-11-04",
  usdPerMillionTokens: { uncachedInput: 0.042, cachedInput: 0, cacheWrite: 0, output: 0 },
  sources: ["https://help.typesafe.ai/articles/5187202169-how-much-does-jev-cost"],
  basis: "All input tokens at the published Jev rate; cache billing categories are not applicable, not provider cache telemetry; not an invoice",
};

interface ModelConfig {
  api: "decisions" | "systemone" | "responses";
  requestModel: string;
  provider: "openai" | "jev";
  endpoint: string;
  keyEnvironment: string;
  pricing: typeof PRICING;
}

const JEV_SERVICE = { api: "systemone", provider: "jev", endpoint: "https://api.typesafe.ai/v1/systemone", keyEnvironment: "TYPESAFE_API_KEY" } satisfies Omit<ModelConfig, "pricing" | "requestModel">;
const RESPONSES_SERVICE = { api: "responses", provider: "openai", endpoint: "https://api.openai.com/v1/responses", keyEnvironment: "OPENAI_API_KEY" } satisfies Omit<ModelConfig, "pricing" | "requestModel">;
export const MODELS = {
  "gpt-6-luna": { api: "decisions", requestModel: "gpt-6-luna", provider: "openai", endpoint: "https://api.openai.com/v1/decisions", keyEnvironment: "OPENAI_API_KEY", pricing: PRICING },
  "jev-1.13.0": { ...JEV_SERVICE, requestModel: "jev-1.13.0", pricing: { ...JEV_PRICING, model: "jev-1.13.0" } },
  "jev-1.14.0": { ...JEV_SERVICE, requestModel: "jev-1.14.0", pricing: { ...JEV_PRICING, model: "jev-1.14.0" } },
  "responses:gpt-6-luna": { ...RESPONSES_SERVICE, requestModel: "gpt-6-luna", pricing: {
    ...PRICING, usdPerMillionTokens: { ...PRICING.usdPerMillionTokens, output: 0.50 },
    sources: ["https://developers.openai.com/api/docs/models/gpt-6-luna"],
  } },
  "responses:gpt-6.1-terra": { ...RESPONSES_SERVICE, requestModel: "gpt-6.1-terra", pricing: {
    ...PRICING, model: "gpt-6.1-terra", usdPerMillionTokens: { uncachedInput: 1, cachedInput: 0.05, cacheWrite: 1.25, output: 5 },
    sources: ["https://developers.openai.com/api/docs/models/gpt-6.1-terra"],
  } },
} satisfies Record<string, ModelConfig>;
export type ModelName = keyof typeof MODELS;

function isModelName(value: string): value is ModelName {
  return Object.hasOwn(MODELS, value);
}

interface MeasuredTokens {
  inputTokens: number;
  cachedTokens: number;
  cacheWriteTokens: number;
  outputTokens: number;
  reasoningTokens: number | null;
}

type Usage =
  | (MeasuredTokens & { kind: "known"; estimatedCostUsd: number })
  | (MeasuredTokens & { kind: "unpriced"; reason: "model" })
  | { kind: "unknown"; reason: "missing" | "invalid" | "transport" };

type RequestOutcome =
  | { kind: "answer"; choice: string; probabilities?: Record<string, number> }
  | { kind: "failure"; category: "http" | "transport" | "parse" | "answer" | "model" };

type CaseIdentity = { task: "pacman"; seed: number } | { task: "rca"; split: "synthetic" | "real"; seed: number; tick: number };

export interface RequestRecord {
  index: number;
  case: CaseIdentity;
  httpStatus: number | null;
  requestId: string | null;
  responseId: string | null;
  responseModel: string | null;
  latencyMs: number;
  usage: Usage;
  outcome: RequestOutcome;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function object(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function tokenCount(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

export function parseUsage(value: unknown, model: ModelName = "gpt-6-luna"): Usage {
  if (value === undefined || value === null) return { kind: "unknown", reason: "missing" };
  const usage = object(value);
  const details = object(usage?.input_tokens_details);
  const input = usage?.input_tokens;
  const config = MODELS[model];
  const cached = config.provider === "jev" ? 0 : details?.cached_tokens;
  const written = config.provider === "jev" ? 0 : details?.cache_write_tokens;
  const output = usage?.output_tokens;
  const reasoning = config.api === "responses" ? object(usage?.output_tokens_details)?.reasoning_tokens : null;
  if (!tokenCount(input) || !tokenCount(cached) || !tokenCount(written) || !tokenCount(output) || cached + written > input) {
    return { kind: "unknown", reason: "invalid" };
  }
  if (config.api === "responses" && (!tokenCount(reasoning) || reasoning > output)) return { kind: "unknown", reason: "invalid" };
  if (reasoning !== null && !tokenCount(reasoning)) return { kind: "unknown", reason: "invalid" };
  if (config.provider === "openai" && details?.cache_write_tokens_details !== undefined) {
    const ttl = object(details.cache_write_tokens_details);
    const short = ttl?.ttl_30m_tokens;
    const long = ttl?.ttl_12h_tokens;
    if (!tokenCount(short) || !tokenCount(long) || short + long !== written || long !== 0) return { kind: "unknown", reason: "invalid" };
  }
  const rates = config.pricing.usdPerMillionTokens;
  return {
    kind: "known", inputTokens: input, cachedTokens: cached, cacheWriteTokens: written, outputTokens: output, reasoningTokens: reasoning,
    estimatedCostUsd: ((input - cached - written) * rates.uncachedInput + cached * rates.cachedInput + written * rates.cacheWrite + output * rates.output) / 1_000_000,
  };
}

function answerBody(value: unknown): DecisionsBody {
  const body = object(value);
  if (!body || !Array.isArray(body.answers)) throw new Error("Invalid answer body");
  const answers: NonNullable<DecisionsBody["answers"]> = [];
  for (const raw of body.answers) {
    const answer = object(raw);
    if (!answer || typeof answer.type !== "string") throw new Error("Invalid answer");
    if (answer.name !== undefined && typeof answer.name !== "string") throw new Error("Invalid answer name");
    if (answer.choice !== undefined && typeof answer.choice !== "string") throw new Error("Invalid choice");
    if (answer.confidence !== undefined && (typeof answer.confidence !== "number" || !Number.isFinite(answer.confidence))) throw new Error("Invalid confidence");
    const probabilities: { value: string; probability: number }[] = [];
    if (answer.probabilities !== undefined) {
      if (!Array.isArray(answer.probabilities)) throw new Error("Invalid probabilities");
      for (const rawProbability of answer.probabilities) {
        const probability = object(rawProbability);
        if (!probability || typeof probability.value !== "string" || typeof probability.probability !== "number" || !Number.isFinite(probability.probability) || probability.probability < 0 || probability.probability > 1) throw new Error("Invalid probability");
        probabilities.push({ value: probability.value, probability: probability.probability });
      }
    }
    answers.push({ type: answer.type, name: answer.name, choice: answer.choice, confidence: answer.confidence, probabilities });
  }
  return { answers };
}

function jevAnswerBody(value: unknown, keys: string[]): DecisionsBody {
  const answers = object(object(value)?.answers);
  const move = object(answers?.move);
  const probabilities = object(move?.probabilities);
  if (!move || !probabilities || Object.keys(probabilities).length !== keys.length || !keys.every(key => Object.hasOwn(probabilities, key))) {
    throw new Error("Incomplete Jev choice probabilities");
  }
  const parsed = answerBody({ answers: [{ ...move, name: "move", probabilities: Object.entries(probabilities).map(([value, probability]) => ({ value, probability })) }] });
  const total = parsed.answers?.[0]?.probabilities?.reduce((sum, p) => sum + p.probability, 0) ?? 0;
  if (Math.abs(total - 1) > 0.001) throw new Error("Invalid Jev probability total");
  return parsed;
}

function buildResponsesRequest(model: string, input: ChoiceInput) {
  return {
    model, instructions: `${input.instructions}\n\nChoice criteria:\n${JSON.stringify(input.criteria)}`,
    input: typeof input.state === "string" ? input.state : JSON.stringify(input.state),
    reasoning: { effort: "none" }, service_tier: "default", store: false, max_output_tokens: 128,
    text: { format: { type: "json_schema", name: "decision", strict: true, schema: {
      type: "object", properties: { choice: { type: "string", enum: input.keys } },
      required: ["choice"], additionalProperties: false,
    } } },
  };
}

function responsesAnswer(value: unknown, keys: string[]): Pick<PolicyDecision, "choice" | "probabilities"> {
  const body = object(value);
  if (body?.status !== "completed" || body.error != null || !Array.isArray(body.output)) throw new Error("Incomplete Responses answer");
  const texts: string[] = [];
  for (const rawItem of body.output) {
    const item = object(rawItem);
    if (item?.type === "reasoning") continue;
    if (item?.type !== "message" || item.role !== "assistant" || item.status !== "completed" || !Array.isArray(item.content)) throw new Error("Invalid Responses output");
    for (const rawContent of item.content) {
      const content = object(rawContent);
      if (content?.type !== "output_text" || typeof content.text !== "string") throw new Error("Responses answer refused or invalid");
      texts.push(content.text);
    }
  }
  if (texts.length !== 1) throw new Error("Expected one Responses choice");
  const choice = object(JSON.parse(texts[0]));
  if (!choice || Object.keys(choice).length !== 1 || typeof choice.choice !== "string" || !keys.includes(choice.choice)) throw new Error("Invalid Responses choice");
  return { choice: choice.choice };
}

export class EvaluationStopped extends Error {}

export function createMeteredClient(options: {
  apiKey: string;
  model?: ModelName;
  record: (record: RequestRecord) => void;
  fetchFn?: typeof fetch;
  maxRequests?: number;
  maxCostUsd?: number;
  maxConsecutiveFailures?: number;
  timeoutMs?: number;
}) {
  const model = options.model ?? "gpt-6-luna";
  const config = MODELS[model];
  const records: RequestRecord[] = [];
  let consecutiveFailures = 0;
  let persistenceFailed = false;
  let inFlight = false;
  let modelMismatch = false;
  const totals = { inputTokens: 0, cachedTokens: 0, cacheWriteTokens: 0, outputTokens: 0, reasoningTokens: config.api === "responses" ? 0 : null, estimatedCostUsd: 0, unknownUsageRequests: 0 };
  function assertCanContinue(): void {
    if (persistenceFailed) throw new EvaluationStopped("Request ledger could not be saved");
    if (modelMismatch) throw new EvaluationStopped("Response model did not match the pinned model");
    if (records.length >= (options.maxRequests ?? 10_000)) throw new EvaluationStopped("Request limit reached");
    if (totals.estimatedCostUsd >= (options.maxCostUsd ?? 1)) throw new EvaluationStopped("Observed cost threshold reached");
    if (consecutiveFailures >= (options.maxConsecutiveFailures ?? 5)) throw new EvaluationStopped("Consecutive failure limit reached");
  }
  return {
    assertCanContinue,
    requestCount: () => records.length,
    summary() {
      const latencies = records.filter(r => r.outcome.kind === "answer").map(r => r.latencyMs);
      const failures = records.filter(r => r.outcome.kind === "failure").length;
      return {
        requests: records.length, attemptedRequests: records.length + Number(inFlight), inFlightRequests: Number(inFlight), successfulRequests: records.length - failures, failedRequests: failures,
        ...totals, costComplete: totals.unknownUsageRequests === 0 && !inFlight,
        latencyP50Ms: latencies.length ? percentile(latencies, 0.5) : null,
        latencyP90Ms: latencies.length ? percentile(latencies, 0.9) : null,
        latencyMinMs: latencies.length ? Math.min(...latencies) : null,
        latencyMaxMs: latencies.length ? Math.max(...latencies) : null,
        responseModels: [...new Set(records.flatMap(r => r.responseModel ? [r.responseModel] : []))],
      };
    },
    async ask(input: ChoiceInput, identity: CaseIdentity): Promise<PolicyDecision> {
      assertCanContinue();
      const request: RequestRecord = {
        index: records.length + 1, case: identity, httpStatus: null, requestId: null,
        responseId: null, responseModel: null, latencyMs: 0,
        usage: { kind: "unknown", reason: "transport" }, outcome: { kind: "failure", category: "transport" },
      };
      inFlight = true;
      const started = performance.now();
      try {
        const body = config.api === "responses" ? buildResponsesRequest(config.requestModel, input) : config.api === "decisions" ? buildDecisionsRequest(config.requestModel, input) : {
          model: config.requestModel, state: input.state,
          questions: { move: { type: "choice", instructions: input.instructions, criteria: input.criteria } },
        };
        const response = await (options.fetchFn ?? fetch)(config.endpoint, {
          method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${options.apiKey}` },
          body: JSON.stringify(body), signal: AbortSignal.timeout(options.timeoutMs ?? 30_000),
        });
        request.httpStatus = response.status;
        request.requestId = response.headers.get("x-request-id");
        const text = await response.text();
        request.latencyMs = performance.now() - started;
        request.outcome = { kind: "failure", category: response.ok ? "parse" : "http" };
        request.usage = { kind: "unknown", reason: "missing" };
        let responseBody: unknown;
        try { responseBody = JSON.parse(text); } catch { throw new Error("Unreadable Decisions response"); }
        const fields = object(responseBody);
        request.usage = parseUsage(fields?.usage, model);
        request.responseId = typeof fields?.id === "string" ? fields.id : null;
        request.responseModel = typeof fields?.model === "string" ? fields.model : null;
        if (!response.ok) throw new Error(`Decisions HTTP ${response.status}`);
        if (request.responseModel !== config.requestModel) {
          modelMismatch = true;
          if (request.usage.kind === "known") {
            const usage = request.usage;
            request.usage = {
              kind: "unpriced", reason: "model", inputTokens: usage.inputTokens, cachedTokens: usage.cachedTokens,
              cacheWriteTokens: usage.cacheWriteTokens, outputTokens: usage.outputTokens, reasoningTokens: usage.reasoningTokens,
            };
          }
          request.outcome = { kind: "failure", category: "model" };
          throw new EvaluationStopped("Response model did not match the pinned model");
        }
        let answer: Pick<PolicyDecision, "choice" | "probabilities" | "confidence">;
        if (config.api === "responses") {
          request.outcome = { kind: "failure", category: "answer" };
          answer = responsesAnswer(responseBody, input.keys);
        } else {
          const parsedBody = config.api === "decisions" ? answerBody(responseBody) : jevAnswerBody(responseBody, input.keys);
          request.outcome = { kind: "failure", category: "answer" };
          answer = parseDecisionsAnswer(parsedBody, input.keys);
        }
        request.outcome = { kind: "answer", choice: answer.choice, probabilities: answer.probabilities };
        return { ...answer, inputTokens: request.usage.kind === "known" ? request.usage.inputTokens : undefined, latencyMs: request.latencyMs };
      } catch (error) {
        if (!request.latencyMs) request.latencyMs = performance.now() - started;
        if (error instanceof EvaluationStopped) throw error;
        if (error instanceof DecisionsAnswerError) throw new Error("Decisions response refused or returned an invalid choice");
        throw new Error(request.httpStatus === null ? "Decisions transport failed" : `Decisions request failed (${request.outcome.kind === "failure" ? request.outcome.category : "unknown"}, HTTP ${request.httpStatus})`);
      } finally {
        inFlight = false;
        records.push(request);
        if (request.usage.kind !== "unknown") {
          totals.inputTokens += request.usage.inputTokens;
          totals.cachedTokens += request.usage.cachedTokens;
          totals.cacheWriteTokens += request.usage.cacheWriteTokens;
          totals.outputTokens += request.usage.outputTokens;
          if (totals.reasoningTokens !== null && request.usage.reasoningTokens !== null) totals.reasoningTokens += request.usage.reasoningTokens;
        }
        if (request.usage.kind === "known") totals.estimatedCostUsd += request.usage.estimatedCostUsd;
        else totals.unknownUsageRequests++;
        consecutiveFailures = request.outcome.kind === "failure" ? consecutiveFailures + 1 : 0;
        try { options.record(request); } catch {
          persistenceFailed = true;
          throw new EvaluationStopped("Request ledger could not be saved");
        }
      }
    },
  };
}

export interface RcaCase { input: ChoiceInput; label: string; seed: number; tick: number }

export function parseRcaCases(text: string): RcaCase[] {
  const identities = new Set<string>();
  const cases = text.split(/\r?\n/).filter(line => line.trim()).map((line, index): RcaCase => {
    const row = object(JSON.parse(line));
    const fail = () => new Error(`Invalid RCA row ${index + 1}`);
    if (!row || !Array.isArray(row.options) || row.options.length !== 23 || !row.options.every((key): key is string => typeof key === "string" && key.length > 0)) throw fail();
    const keys: string[] = row.options;
    if (new Set(keys).size !== keys.length || typeof row.instructions !== "string" || !row.instructions.trim()) throw fail();
    if (typeof row.state !== "string" && !object(row.state)) throw fail();
    if (typeof row.label !== "string" || !keys.includes(row.label) || !tokenCount(row.seed) || !tokenCount(row.tick)) throw fail();
    const rawCriteria = object(row.criteria);
    if (!rawCriteria || Object.keys(rawCriteria).length !== keys.length) throw fail();
    const criteria: Record<string, string | null> = {};
    for (const key of keys) {
      const value = rawCriteria[key];
      if (value !== null && typeof value !== "string") throw fail();
      criteria[key] = value;
    }
    const identity = `${row.seed}:${row.tick}`;
    if (identities.has(identity)) throw new Error(`Duplicate RCA seed/tick at row ${index + 1}`);
    identities.add(identity);
    return { input: { state: row.state, instructions: row.instructions, criteria, keys }, label: row.label, seed: row.seed, tick: row.tick };
  });
  if (!cases.length) throw new Error("RCA input is empty");
  return cases;
}

type RcaOutcome = { seed: number; tick: number; label: string; prediction: string | null };

export function scoreRca(cases: RcaCase[], outcomes: RcaOutcome[]) {
  const predictions = new Map(outcomes.map(row => [`${row.seed}:${row.tick}`, row.prediction]));
  const classes = new Map<string, { total: number; correct: number; predicted: number }>();
  let correct = 0;
  for (const row of cases) {
    const counts = classes.get(row.label) ?? { total: 0, correct: 0, predicted: 0 };
    counts.total++;
    const prediction = predictions.get(`${row.seed}:${row.tick}`);
    if (prediction === row.label) { correct++; counts.correct++; }
    classes.set(row.label, counts);
  }
  for (const row of outcomes) {
    if (row.prediction !== null) {
      const counts = classes.get(row.prediction) ?? { total: 0, correct: 0, predicted: 0 };
      counts.predicted++;
      classes.set(row.prediction, counts);
    }
  }
  const perClass = [...classes].map(([label, counts]) => ({ label, ...counts, recall: counts.total ? counts.correct / counts.total : null }));
  const present = perClass.filter(c => c.total > 0);
  const productBug = classes.get("product_bug") ?? { total: 0, correct: 0, predicted: 0 };
  const majority = present.reduce((best, c) => c.total > best.total ? c : best);
  return {
    total: cases.length, completed: outcomes.length, correct, accuracy: correct / cases.length,
    failed: outcomes.filter(row => row.prediction === null).length, perClass,
    macroRecall: present.reduce((sum, c) => sum + (c.recall ?? 0), 0) / present.length,
    majorityBaseline: { label: majority.label, correct: majority.total, total: cases.length, accuracy: majority.total / cases.length },
    productBug: { ...productBug, accuracy: productBug.total ? productBug.correct / productBug.total : null },
    otherClasses: { total: cases.length - productBug.total, correct: correct - productBug.correct, accuracy: cases.length > productBug.total ? (correct - productBug.correct) / (cases.length - productBug.total) : null },
  };
}

function positiveInteger(value: string, name: string, minimum = 1): number {
  const result = Number(value);
  if (!Number.isSafeInteger(result) || result < minimum) throw new Error(`${name} must be an integer >= ${minimum}`);
  return result;
}

async function main() {
  const { values } = parseArgs({ options: {
    model: { type: "string", default: "gpt-6-luna" },
    task: { type: "string" }, input: { type: "string" }, split: { type: "string" }, out: { type: "string" },
    games: { type: "string", default: "10" }, seed: { type: "string", default: "100" }, "max-seconds": { type: "string", default: "300" },
    "max-requests": { type: "string", default: "10000" }, "max-cost-usd": { type: "string", default: "1" },
    "max-consecutive-failures": { type: "string", default: "5" },
  } });
  if (values.task !== "pacman" && values.task !== "rca") throw new Error("--task must be pacman or rca");
  const task = values.task;
  if (!values.out) throw new Error("--out is required and must name a new directory");
  if (!isModelName(values.model)) throw new Error(`--model must be one of ${Object.keys(MODELS).join(", ")}`);
  const model = values.model;
  const config = MODELS[model];
  const apiKey = process.env[config.keyEnvironment];
  if (!apiKey) throw new Error(`Set ${config.keyEnvironment} in the environment`);
  const gamesRequested = positiveInteger(values.games, "games");
  const seed0 = positiveInteger(values.seed, "seed", 0);
  const maxSeconds = positiveInteger(values["max-seconds"], "max-seconds");
  if (task === "pacman" && (seed0 < 100 || seed0 + gamesRequested > 200)) throw new Error("Pac-Man evaluation seeds must stay in 100-199");
  const maxRequests = positiveInteger(values["max-requests"], "max-requests");
  const maxConsecutiveFailures = positiveInteger(values["max-consecutive-failures"], "max-consecutive-failures");
  const maxCostUsd = Number(values["max-cost-usd"]);
  if (!Number.isFinite(maxCostUsd) || maxCostUsd <= 0) throw new Error("max-cost-usd must be positive");
  let dataset: { cases: RcaCase[]; sha256: string; split: "synthetic" | "real" } | null = null;
  if (task === "rca") {
    if (!values.input || (values.split !== "synthetic" && values.split !== "real")) throw new Error("RCA needs --input and --split synthetic|real");
    const bytes = readFileSync(values.input);
    dataset = { cases: parseRcaCases(bytes.toString("utf8")), sha256: createHash("sha256").update(bytes).digest("hex"), split: values.split };
  }
  const out = resolve(values.out);
  mkdirSync(dirname(out), { recursive: true });
  mkdirSync(out);
  const startedAt = new Date().toISOString();
  let status: "running" | "complete" | "aborted" = "running";
  let stopReason: string | null = null;
  const games: GameSummary[] = [];
  const rcaOutcomes: RcaOutcome[] = [];
  const client = createMeteredClient({ apiKey, model, maxRequests, maxCostUsd, maxConsecutiveFailures,
    record: request => appendFileSync(join(out, "requests.jsonl"), JSON.stringify(request) + "\n"),
  });
  const save = () => {
    const summary = {
      status, stopReason, task, model, requestModel: config.requestModel, api: config.api, provider: config.provider, startedAt, updatedAt: new Date().toISOString(), pricing: config.pricing,
      requestSettings: config.api === "responses" ? { reasoningEffort: "none", serviceTier: "default", maxOutputTokens: 128, store: false, cacheWriteTtl: "30m (documented default)" } : null,
      hardware: { client: `${platform()} ${arch()}`, logicalCpus: cpus().length, cpu: cpus()[0]?.model ?? "unknown", node: process.version, inference: `${config.provider === "openai" ? "OpenAI" : "TypeSafe"} hosted; hardware not disclosed` },
      limits: { maxRequests, maxCostUsd, maxConsecutiveFailures, costLimitIsStopAfterResponse: true, retries: 0 },
      latencyDefinition: "HTTP request through full response body; excludes parsing and ledger I/O; successful responses only",
      usage: client.summary(),
      ...(dataset ? { protocol: { encoder: "rca-facts", split: dataset.split, datasetSha256: dataset.sha256, choices: 23, total: dataset.cases.length }, results: scoreRca(dataset.cases, rcaOutcomes) } : {
        protocol: { encoder: "features", clock: "realtime", speed: 1, seed0, gamesRequested, maxSeconds }, games,
        results: {
          completedGames: games.length,
          meanPellets: games.length ? games.reduce((s, g) => s + g.pellets, 0) / games.length : null,
          pelletsMin: games.length ? Math.min(...games.map(g => g.pellets)) : null,
          pelletsMax: games.length ? Math.max(...games.map(g => g.pellets)) : null,
          meanSeconds: games.length ? games.reduce((s, g) => s + g.secondsPlayed, 0) / games.length : null,
          meanGameLatencyP50Ms: games.length ? games.reduce((s, g) => s + g.latencyP50, 0) / games.length : null,
          stale: games.reduce((s, g) => s + g.stale, 0), failures: games.reduce((s, g) => s + g.failures, 0),
        },
      }),
    };
    writeFileSync(join(out, "summary.tmp"), JSON.stringify(summary, null, 2));
    renameSync(join(out, "summary.tmp"), join(out, "summary.json"));
    return summary;
  };
  const abort = () => { status = "aborted"; stopReason = "Interrupted by signal"; save(); process.exit(130); };
  process.once("SIGINT", abort);
  process.once("SIGTERM", abort);
  save();
  try {
    if (dataset) {
      for (const row of dataset.cases) {
        client.assertCanContinue();
        let prediction: string | null = null;
        try { prediction = (await client.ask(row.input, { task: "rca", split: dataset.split, seed: row.seed, tick: row.tick })).choice; }
        catch (error) { if (error instanceof EvaluationStopped) throw error; }
        const outcome = { seed: row.seed, tick: row.tick, label: row.label, prediction };
        rcaOutcomes.push(outcome);
        appendFileSync(join(out, "outcomes.jsonl"), JSON.stringify(outcome) + "\n");
        save();
        if (rcaOutcomes.length % 25 === 0) process.stdout.write(JSON.stringify({ completed: rcaOutcomes.length, total: dataset.cases.length, ...client.summary() }) + "\n");
      }
    } else {
      for (let i = 0; i < gamesRequested; i++) {
        const seed = seed0 + i;
        client.assertCanContinue();
        const game = await playGame({ seed, encoder: ENCODERS.features, clock: "realtime", speed: 1, maxSeconds,
          policy: { name: `${config.provider}:${model}`, decide: input => client.ask(input, { task: "pacman", seed }) },
          onDecision: event => {
            appendFileSync(join(out, "decisions.jsonl"), JSON.stringify({ seed, askedTick: event.askedTick, answeredTick: event.answeredTick, choice: event.decision?.choice ?? null, stale: event.stale, failed: !!event.error }) + "\n");
            if (client.requestCount() % 100 === 0) save();
            client.assertCanContinue();
          },
        });
        games.push(game);
        save();
        process.stdout.write(JSON.stringify({ game, usage: client.summary() }) + "\n");
      }
    }
    status = "complete";
  } catch (error) {
    status = "aborted";
    stopReason = error instanceof EvaluationStopped ? error.message : "Evaluation could not finish";
    process.exitCode = 1;
  } finally {
    process.removeListener("SIGINT", abort);
    process.removeListener("SIGTERM", abort);
    const summary = save();
    process.stdout.write(JSON.stringify(summary) + "\n");
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    process.stderr.write((error instanceof Error ? error.message : "Evaluation failed") + "\n");
    process.exitCode = 1;
  });
}
