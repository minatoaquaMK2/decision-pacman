import { describe, expect, it } from "vitest";
import { buildDecisionsRequest } from "../src/agent/decisions.ts";
import { createMeteredClient, EvaluationStopped, parseRcaCases, parseUsage, scoreRca, type RequestRecord } from "../scripts/eval_openai.ts";

const input = { state: "Facts only", instructions: "Choose", criteria: { left: null, right: null }, keys: ["left", "right"] };
const identity = { task: "pacman", seed: 100 } satisfies Parameters<ReturnType<typeof createMeteredClient>["ask"]>[1];
const usage = { input_tokens: 1000, input_tokens_details: { cached_tokens: 200, cache_write_tokens: 300 }, output_tokens: 0 };
const body = {
  id: "decision-test", model: "gpt-6-luna",
  answers: [{ type: "choice", name: "move", choice: "left", probabilities: [{ value: "left", probability: 0.7 }, { value: "right", probability: 0.3 }] }], usage,
};
const response = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { "x-request-id": "request-test" } });

function rcaRow(label = "product_bug", tick = 1) {
  const options = ["product_bug", ...Array.from({ length: 22 }, (_, i) => `cause_${i}`)];
  return {
    state: { symptoms: "An observed failure" }, instructions: "Choose the cause", options,
    criteria: Object.fromEntries(options.map(k => [k, `Description of ${k}`])),
    label, seed: 1000, tick, reason: "SECRET_GROUND_TRUTH", values: { product_bug: 1 }, source: "private source",
  };
}

describe("Decisions usage accounting", () => {
  it("charges each input token exactly once with cache discounts and writes", () => {
    expect(parseUsage(usage)).toEqual({ kind: "known", inputTokens: 1000, cachedTokens: 200, cacheWriteTokens: 300, outputTokens: 0, reasoningTokens: null, estimatedCostUsd: (500 * 0.10 + 200 * 0.01 + 300 * 0.125) / 1_000_000 });
  });

  it("keeps absent and inconsistent usage unknown rather than claiming zero", () => {
    expect(parseUsage(undefined)).toEqual({ kind: "unknown", reason: "missing" });
    expect(parseUsage({ input_tokens: 1000, output_tokens: 0 }).kind).toBe("unknown");
    expect(parseUsage({ ...usage, input_tokens: 400 }).kind).toBe("unknown");
    expect(parseUsage({ ...usage, input_tokens: -1 }).kind).toBe("unknown");
  });

  it("records IDs, legal results, and measured usage without request facts or credentials", async () => {
    const records: RequestRecord[] = [];
    const client = createMeteredClient({ apiKey: "secret-test-key", record: r => records.push(r), fetchFn: async () => response(body) });
    const answer = await client.ask(input, identity);
    expect(answer.choice).toBe("left");
    expect(answer.inputTokens).toBe(1000);
    expect(records).toHaveLength(1);
    expect(records[0]).toMatchObject({ requestId: "request-test", responseId: "decision-test", responseModel: "gpt-6-luna", usage: { kind: "known" }, outcome: { kind: "answer", choice: "left" } });
    expect(JSON.stringify(records)).not.toContain("secret-test-key");
    expect(JSON.stringify(records)).not.toContain("Facts only");
    expect(client.summary()).toMatchObject({ requests: 1, successfulRequests: 1, costComplete: true, inputTokens: 1000 });
  });

  it("counts billable refusals before rejecting their answer", async () => {
    const records: RequestRecord[] = [];
    const client = createMeteredClient({ apiKey: "test", record: r => records.push(r), fetchFn: async () => response({ ...body, answers: [{ type: "refusal", name: "move" }] }) });
    await expect(client.ask(input, identity)).rejects.toThrow(/refused/);
    expect(records[0].usage.kind).toBe("known");
    expect(client.summary()).toMatchObject({ failedRequests: 1, inputTokens: 1000, costComplete: true });
    expect(client.summary().estimatedCostUsd).toBeGreaterThan(0);
  });

  it("marks cost incomplete while a submitted request has no response yet", async () => {
    let complete = (_response: Response) => {};
    const pending = new Promise<Response>(resolve => { complete = resolve; });
    const client = createMeteredClient({ apiKey: "test", record: () => {}, fetchFn: () => pending });
    const answer = client.ask(input, identity);
    expect(client.summary()).toMatchObject({ attemptedRequests: 1, requests: 0, inFlightRequests: 1, costComplete: false });
    complete(response(body));
    await answer;
    expect(client.summary()).toMatchObject({ attemptedRequests: 1, requests: 1, inFlightRequests: 0, costComplete: true });
  });

  it("counts malformed billable answers, and reports HTTP failures with unknown usage", async () => {
    const records: RequestRecord[] = [];
    let calls = 0;
    const client = createMeteredClient({ apiKey: "test", record: r => records.push(r), fetchFn: async () => ++calls === 1 ? response({ model: "gpt-6-luna", usage, answers: "broken" }) : response({ error: { message: "private body" } }, 429) });
    await expect(client.ask(input, identity)).rejects.toThrow(/parse/);
    await expect(client.ask(input, identity)).rejects.toThrow(/HTTP 429/);
    expect(client.summary()).toMatchObject({ requests: 2, failedRequests: 2, unknownUsageRequests: 1, costComplete: false, inputTokens: 1000 });
    expect(JSON.stringify(records)).not.toContain("private body");
  });

  it("does not retry or issue another request after the failure cutoff", async () => {
    let calls = 0;
    const client = createMeteredClient({ apiKey: "test", record: () => {}, maxConsecutiveFailures: 2, fetchFn: async () => { calls++; throw new Error("private network error"); } });
    await expect(client.ask(input, identity)).rejects.toThrow("Decisions transport failed");
    await expect(client.ask(input, identity)).rejects.toThrow("Decisions transport failed");
    expect(() => client.assertCanContinue()).toThrow(EvaluationStopped);
    await expect(client.ask(input, identity)).rejects.toThrow("failure limit");
    expect(calls).toBe(2);
    expect(client.summary()).toMatchObject({ requests: 2, unknownUsageRequests: 2 });
  });

  it("stops before another request after reaching spend or request limits", async () => {
    for (const limits of [{ maxRequests: 1 }, { maxCostUsd: 0.000001 }]) {
      let calls = 0;
      const client = createMeteredClient({ ...limits, apiKey: "test", record: () => {}, fetchFn: async () => { calls++; return response(body); } });
      await client.ask(input, identity);
      await expect(client.ask(input, identity)).rejects.toThrow(EvaluationStopped);
      expect(calls).toBe(1);
    }
  });

  it("stops spending when the request ledger cannot be persisted", async () => {
    let calls = 0;
    const client = createMeteredClient({ apiKey: "test", record: () => { throw new Error("disk full"); }, fetchFn: async () => { calls++; return response(body); } });
    await expect(client.ask(input, identity)).rejects.toThrow(/ledger/);
    await expect(client.ask(input, identity)).rejects.toThrow(/ledger/);
    expect(calls).toBe(1);
  });
});

describe("RCA evaluation", () => {
  it("projects only facts and legal options into the model request", () => {
    const [row] = parseRcaCases(JSON.stringify(rcaRow()));
    const request = JSON.stringify(buildDecisionsRequest("gpt-6-luna", row.input));
    expect(request).toContain("An observed failure");
    expect(request).not.toContain("SECRET_GROUND_TRUTH");
    expect(request).not.toContain("private source");
    expect(request).not.toContain('"label"');
    expect(row.input.keys).toHaveLength(23);
  });

  it("rejects duplicate records, wrong option counts, and illegal labels before calls", () => {
    const row = rcaRow();
    expect(() => parseRcaCases([row, row].map(r => JSON.stringify(r)).join("\n"))).toThrow(/Duplicate/);
    expect(() => parseRcaCases(JSON.stringify({ ...row, options: ["product_bug"] }))).toThrow(/Invalid/);
    expect(() => parseRcaCases(JSON.stringify({ ...row, label: "not_a_class" }))).toThrow(/Invalid/);
  });

  it("keeps failures in accuracy and shows class imbalance and macro recall", () => {
    const cases = parseRcaCases([rcaRow("product_bug", 1), rcaRow("product_bug", 2), rcaRow("cause_0", 3)].map(r => JSON.stringify(r)).join("\n"));
    const outcomes = cases.map((row, index) => ({ seed: row.seed, tick: row.tick, label: row.label, prediction: index === 2 ? null : "product_bug" }));
    const result = scoreRca(cases, outcomes);
    expect(result).toMatchObject({ total: 3, completed: 3, correct: 2, failed: 1, accuracy: 2 / 3, macroRecall: 0.5, majorityBaseline: { label: "product_bug", accuracy: 2 / 3 }, productBug: { correct: 2, total: 2 }, otherClasses: { correct: 0, total: 1 } });
  });
});

describe("pinned Jev evaluations", () => {
  const jevUsage = { input_tokens: 1000, output_tokens: 31 };
  const jevBody = (model: string, probabilities: Record<string, number> = { left: 0.7, right: 0.3 }, choice = "left") => ({
    model, usage: jevUsage, answers: { move: { type: "choice", choice, confidence: 0.4, probabilities } },
  });

  it("charges all Jev input tokens at its flat rate and does not charge output", () => {
    for (const model of ["jev-1.13.0", "jev-1.14.0"] as const) {
      expect(parseUsage(jevUsage, model)).toEqual({ kind: "known", inputTokens: 1000, cachedTokens: 0, cacheWriteTokens: 0, outputTokens: 31, reasoningTokens: null, estimatedCostUsd: 0.000042 });
      expect(parseUsage({ input_tokens: 1000 }, model).kind).toBe("unknown");
      expect(parseUsage({ input_tokens: -1, output_tokens: 31 }, model).kind).toBe("unknown");
    }
    expect(parseUsage(jevUsage).kind).toBe("unknown");
  });

  it("sends the same factual choices through System One for each pinned version", async () => {
    const [row] = parseRcaCases(JSON.stringify(rcaRow()));
    const probabilities = Object.fromEntries(row.input.keys.map(key => [key, key === "product_bug" ? 1 : 0]));
    for (const model of ["jev-1.13.0", "jev-1.14.0"] as const) {
      let sent = "";
      const records: RequestRecord[] = [];
      const client = createMeteredClient({ model, apiKey: "test-jev-key", record: r => records.push(r), fetchFn: async (url, init) => {
        expect(url).toBe("https://api.typesafe.ai/v1/systemone");
        expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer test-jev-key");
        sent = String(init?.body);
        expect(JSON.parse(sent)).toEqual({ model, state: row.input.state, questions: { move: { type: "choice", instructions: row.input.instructions, criteria: row.input.criteria } } });
        return response(jevBody(model, probabilities, "product_bug"));
      } });
      const answer = await client.ask(row.input, { task: "rca", split: "real", seed: row.seed, tick: row.tick });
      expect(answer.choice).toBe("product_bug");
      expect(answer.inputTokens).toBe(1000);
      expect(sent).not.toContain("SECRET_GROUND_TRUTH");
      expect(sent).not.toContain('"label"');
      expect(sent).not.toContain("keep_alive");
      expect(JSON.stringify(records)).not.toContain("test-jev-key");
      expect(client.summary()).toMatchObject({ requests: 1, failedRequests: 0, costComplete: true, estimatedCostUsd: 0.000042, responseModels: [model] });
    }
  });

  it("retains usage for an illegal choice or malformed Jev probabilities", async () => {
    for (const reply of [
      jevBody("jev-1.13.0", { left: 0.7, right: 0.3 }, "not-legal"),
      jevBody("jev-1.13.0", { left: 1 }),
      jevBody("jev-1.13.0", { left: 0.7, right: 0.7 }),
      jevBody("jev-1.13.0", { left: -0.1, right: 1.1 }),
    ]) {
      const client = createMeteredClient({ model: "jev-1.13.0", apiKey: "test", record: () => {}, fetchFn: async () => response(reply) });
      await expect(client.ask(input, identity)).rejects.toThrow();
      expect(client.summary()).toMatchObject({ failedRequests: 1, costComplete: true, estimatedCostUsd: 0.000042 });
    }
  });

  it("retains tokens with unresolved cost and stops after a substituted Jev model", async () => {
    let calls = 0;
    const records: RequestRecord[] = [];
    const client = createMeteredClient({ model: "jev-1.13.0", apiKey: "test", record: r => records.push(r), fetchFn: async () => {
      calls++;
      return response(jevBody("jev-1.14.0"));
    } });
    await expect(client.ask(input, identity)).rejects.toThrow(EvaluationStopped);
    expect(records[0]).toMatchObject({ responseModel: "jev-1.14.0", outcome: { kind: "failure", category: "model" } });
    expect(records[0].usage).toEqual({ kind: "unpriced", reason: "model", inputTokens: 1000, cachedTokens: 0, cacheWriteTokens: 0, outputTokens: 31, reasoningTokens: null });
    expect(client.summary()).toMatchObject({ failedRequests: 1, inputTokens: 1000, outputTokens: 31, estimatedCostUsd: 0, unknownUsageRequests: 1, costComplete: false });
    expect(() => client.assertCanContinue()).toThrow(/pinned model/);
    await expect(client.ask(input, identity)).rejects.toThrow(EvaluationStopped);
    expect(calls).toBe(1);
  });

  it("keeps an HTTP error without usage incomplete and never includes its private text", async () => {
    const records: RequestRecord[] = [];
    const client = createMeteredClient({ model: "jev-1.14.0", apiKey: "test", record: r => records.push(r), fetchFn: async () => response({ error: "private error text" }, 402) });
    await expect(client.ask(input, identity)).rejects.toThrow(/HTTP 402/);
    expect(client.summary()).toMatchObject({ requests: 1, failedRequests: 1, unknownUsageRequests: 1, costComplete: false });
    expect(JSON.stringify(records)).not.toContain("private error text");
  });
});

describe("Responses evaluations", () => {
  const responseUsage = { ...usage, output_tokens: 100, output_tokens_details: { reasoning_tokens: 25 } };
  const responseBody = (model = "gpt-6-luna", text = '{"choice":"left"}') => ({
    id: "response-test", model, status: "completed", usage: responseUsage,
    output: [{ type: "message", role: "assistant", status: "completed", content: [{ type: "output_text", text }] }],
  });

  it("bills output once, including its reasoning subset, at each route's rates", () => {
    expect(parseUsage(responseUsage, "responses:gpt-6-luna")).toEqual({
      kind: "known", inputTokens: 1000, cachedTokens: 200, cacheWriteTokens: 300, outputTokens: 100, reasoningTokens: 25,
      estimatedCostUsd: (500 * 0.1 + 200 * 0.01 + 300 * 0.125 + 100 * 0.5) / 1_000_000,
    });
    expect(parseUsage(responseUsage, "responses:gpt-6.1-terra")).toMatchObject({
      kind: "known", reasoningTokens: 25, estimatedCostUsd: (500 + 200 * 0.05 + 300 * 1.25 + 100 * 5) / 1_000_000,
    });
    expect(parseUsage({ ...responseUsage, output_tokens_details: { reasoning_tokens: 101 } }, "responses:gpt-6-luna").kind).toBe("unknown");
    expect(parseUsage({ ...responseUsage, output_tokens_details: undefined }, "responses:gpt-6-luna").kind).toBe("unknown");
  });

  it("does not silently use the short-cache price for longer or inconsistent writes", () => {
    const withTtl = (short: number, long: number) => ({ ...responseUsage, input_tokens_details: { ...responseUsage.input_tokens_details, cache_write_tokens_details: { ttl_30m_tokens: short, ttl_12h_tokens: long } } });
    expect(parseUsage(withTtl(300, 0), "responses:gpt-6-luna").kind).toBe("known");
    expect(parseUsage(withTtl(200, 100), "responses:gpt-6-luna").kind).toBe("unknown");
    expect(parseUsage(withTtl(299, 0), "responses:gpt-6-luna").kind).toBe("unknown");
  });

  it("sends original facts and criteria with strict legal choices and no hidden labels", async () => {
    const [row] = parseRcaCases(JSON.stringify(rcaRow()));
    for (const route of ["responses:gpt-6-luna", "responses:gpt-6.1-terra"] as const) {
      const actualModel = route.slice("responses:".length);
      const records: RequestRecord[] = [];
      const client = createMeteredClient({ model: route, apiKey: "responses-test-key", record: r => records.push(r), fetchFn: async (url, init) => {
        expect(url).toBe("https://api.openai.com/v1/responses");
        const sent = String(init?.body);
        const request = JSON.parse(sent);
        expect(request.model).toBe(actualModel);
        expect(request.input).toBe(JSON.stringify(row.input.state));
        expect(request.instructions).toContain(row.input.instructions);
        expect(request.instructions).toContain(JSON.stringify(row.input.criteria));
        expect(request).toMatchObject({ reasoning: { effort: "none" }, service_tier: "default", store: false, max_output_tokens: 128, text: { format: { type: "json_schema", strict: true, schema: { properties: { choice: { enum: row.input.keys } }, required: ["choice"], additionalProperties: false } } } });
        expect(sent).not.toContain("SECRET_GROUND_TRUTH");
        expect(sent).not.toContain("private source");
        expect(sent).not.toContain('"label"');
        return response(responseBody(actualModel, '{"choice":"product_bug"}'));
      } });
      const answer = await client.ask(row.input, { task: "rca", split: "real", seed: row.seed, tick: row.tick });
      expect(answer.choice).toBe("product_bug");
      expect(answer.probabilities).toBeUndefined();
      expect(answer.confidence).toBeUndefined();
      expect(client.summary()).toMatchObject({ requests: 1, successfulRequests: 1, reasoningTokens: 25, responseModels: [actualModel], costComplete: true });
      expect(JSON.stringify(records)).not.toContain("responses-test-key");
      expect(JSON.stringify(records)).not.toContain("symptoms");
    }
  });

  it("retains billed usage for refusals, incomplete responses, and malformed choices", async () => {
    const complete = responseBody();
    for (const reply of [
      { ...complete, status: "incomplete" },
      { ...complete, output: [{ type: "message", role: "assistant", status: "completed", content: [{ type: "refusal", refusal: "private refusal" }] }] },
      responseBody("gpt-6-luna", "invalid JSON"),
      responseBody("gpt-6-luna", '{"choice":"not-legal"}'),
      responseBody("gpt-6-luna", '{"choice":"left","extra":true}'),
      { ...complete, output: [...complete.output, ...complete.output] },
    ]) {
      const records: RequestRecord[] = [];
      const client = createMeteredClient({ model: "responses:gpt-6-luna", apiKey: "test", record: r => records.push(r), fetchFn: async () => response(reply) });
      await expect(client.ask(input, identity)).rejects.toThrow();
      expect(client.summary()).toMatchObject({ failedRequests: 1, inputTokens: 1000, outputTokens: 100, reasoningTokens: 25, costComplete: true });
      expect(client.summary().estimatedCostUsd).toBeGreaterThan(0);
      expect(records[0].outcome).toEqual({ kind: "failure", category: "answer" });
      expect(JSON.stringify(records)).not.toContain("private refusal");
    }
  });

  it("keeps substituted model tokens without pricing them at the requested model's rate", async () => {
    let calls = 0;
    const records: RequestRecord[] = [];
    const client = createMeteredClient({ model: "responses:gpt-6.1-terra", apiKey: "test", record: r => records.push(r), fetchFn: async () => { calls++; return response(responseBody()); } });
    await expect(client.ask(input, identity)).rejects.toThrow(EvaluationStopped);
    expect(records[0].usage).toEqual({ kind: "unpriced", reason: "model", inputTokens: 1000, cachedTokens: 200, cacheWriteTokens: 300, outputTokens: 100, reasoningTokens: 25 });
    expect(() => client.assertCanContinue()).toThrow(/pinned model/);
    await expect(client.ask(input, identity)).rejects.toThrow(EvaluationStopped);
    expect(calls).toBe(1);
    expect(client.summary()).toMatchObject({ requests: 1, failedRequests: 1, inputTokens: 1000, cachedTokens: 200, cacheWriteTokens: 300, outputTokens: 100, reasoningTokens: 25, estimatedCostUsd: 0, unknownUsageRequests: 1, costComplete: false });
  });
});
