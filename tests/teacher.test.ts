import { describe, expect, it } from "vitest";
import { parseTeacherReply, TeacherParseError, teacherPolicy, teacherPrompt, type TeacherDecision } from "../src/agent/teacher.ts";
import { createGame, step } from "../src/engine/game.ts";
import { findDecisionPoint } from "../src/engine/decision.ts";
import { ENCODERS } from "../src/encoders/index.ts";

describe("teacher reply parsing", () => {
  const keys = ["up", "left", "back"];

  it("reads the move and reason", () => {
    expect(parseTeacherReply('{"move": "left", "reason": "more pellets"}', keys)).toEqual({ move: "left", reason: "more pellets" });
  });

  it("uses the last JSON answer after any thinking text", () => {
    const text = 'I could go {"move": "up"} maybe... Final: {"move": "back", "reason": "ghost ahead"}';
    expect(parseTeacherReply(text, keys).move).toBe("back");
  });

  it("rejects moves that are not options", () => {
    expect(() => parseTeacherReply('{"move": "down", "reason": "x"}', keys)).toThrow(TeacherParseError);
  });

  it("rejects replies without an answer", () => {
    expect(() => parseTeacherReply("left seems fine", keys)).toThrow(TeacherParseError);
  });
});

describe("teacher prompt", () => {
  it("contains the rules, the student's state, and every option", () => {
    const s = createGame(1);
    while (s.phase !== "playing") step(s);
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    const prompt = teacherPrompt(enc);
    expect(prompt).toContain(enc.instructions);
    expect(prompt).toContain(JSON.stringify(enc.state));
    for (const k of enc.keys) expect(prompt).toContain(`- ${k}`);
  });
});

describe("teacherPrompt with lookahead", () => {
  it("adds one simulated outcome per option", () => {
    const s = createGame(1);
    while (s.phase !== "playing") step(s);
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    const outcomes = Object.fromEntries(enc.keys.map((k) => [k, { dies_after_s: null, pellets: 3, power_pellets: 0, ghosts_eaten: 0, points: 30 }]));
    const prompt = teacherPrompt(enc, { seconds: 5, outcomes });
    expect(prompt).toContain("Lookahead:");
    expect(prompt).toContain(JSON.stringify(outcomes));
    expect(teacherPrompt(enc)).not.toContain("Lookahead:");
  });
});

describe("teacher answer cache", () => {
  it("asks the server once per distinct prompt", async () => {
    let calls = 0;
    const fetchFn = (async () => {
      calls++;
      return new Response(JSON.stringify({ choices: [{ message: { content: '{"move": "back", "reason": "r"}' } }], usage: { completion_tokens: 9 } }));
    }) as unknown as typeof fetch;
    const cache = new Map();
    const policy = teacherPolicy({ baseUrl: "http://teacher.invalid/v1", model: "m", cache, fetchFn });
    const s = createGame(1);
    while (s.phase !== "playing") step(s);
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    const first = (await policy.decide(enc, undefined, s)) as TeacherDecision;
    const second = (await policy.decide(enc, undefined, s)) as TeacherDecision;
    expect(calls).toBe(1);
    expect(first.cached).toBeUndefined();
    expect(second.cached).toBe(true);
    expect(second.choice).toBe(first.choice);
    expect(cache.size).toBe(1);
  });
});

describe("teacher on a hosted API", () => {
  const reply = (usage: Record<string, unknown> = {}) =>
    new Response(JSON.stringify({ choices: [{ message: { content: '{"move": "back", "reason": "r"}' } }], usage }));

  it("sends the key, merges extra body fields, and reads usage", async () => {
    let seen: { headers: Record<string, string>; body: Record<string, unknown> } | undefined;
    const fetchFn = (async (_url: string, init: RequestInit) => {
      seen = { headers: init.headers as Record<string, string>, body: JSON.parse(String(init.body)) };
      return reply({ prompt_tokens: 450, completion_tokens: 36, completion_tokens_details: { reasoning_tokens: 0 }, cost: 0.0004 });
    }) as unknown as typeof fetch;
    const policy = teacherPolicy({
      baseUrl: "http://teacher.invalid/v1",
      model: "m",
      apiKey: "test-key",
      extraBody: { reasoning: { enabled: false } },
      fetchFn,
    });
    const s = createGame(1);
    while (s.phase !== "playing") step(s);
    const d = (await policy.decide(ENCODERS.features.encode(s, findDecisionPoint(s)), undefined, s)) as TeacherDecision;
    expect(seen!.headers.Authorization).toBe("Bearer test-key");
    expect(seen!.body.reasoning).toEqual({ enabled: false });
    expect(seen!.body.chat_template_kwargs).toEqual({ enable_thinking: false });
    expect(d.inputTokens).toBe(450);
    expect(d.outputTokens).toBe(36);
    expect(d.reasoningTokens).toBe(0);
    expect(d.costUsd).toBe(0.0004);
  });

  it("retries after 429 when asked to, and fails at once by default", async () => {
    let calls = 0;
    const fetchFn = (async () => {
      calls++;
      return calls === 1 ? new Response("Rate limit exceeded. Retry after 0s.", { status: 429 }) : reply();
    }) as unknown as typeof fetch;
    const s = createGame(1);
    while (s.phase !== "playing") step(s);
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    await expect(teacherPolicy({ baseUrl: "http://teacher.invalid/v1", model: "m", fetchFn }).decide(enc, undefined, s)).rejects.toThrow(/429/);
    calls = 0;
    const d = (await teacherPolicy({ baseUrl: "http://teacher.invalid/v1", model: "m", maxRetries: 2, retryBaseMs: 1, fetchFn }).decide(
      enc,
      undefined,
      s,
    )) as TeacherDecision;
    expect(calls).toBe(2);
    expect(d.retries).toBe(1);
    expect(d.choice).toBe("back");
  });
});

describe("teacher option probabilities", () => {
  it("reads each option's probability at the move token when asked for logprobs", async () => {
    let body: Record<string, unknown> = {};
    const content = '{"move": "back", "reason": "r"}';
    const tokens = ['{"', "move", '":', ' "', "back", '",'].map((token) => ({ token, logprob: -0.1 }));
    tokens[4] = { token: "back", logprob: Math.log(0.6), top_logprobs: [{ token: "back", logprob: Math.log(0.6) }, { token: "up", logprob: Math.log(0.3) }, { token: "}", logprob: Math.log(0.1) }] } as never;
    const fetchFn = (async (_url: string, init: RequestInit) => {
      body = JSON.parse(String(init.body));
      return new Response(JSON.stringify({ choices: [{ message: { content }, logprobs: { content: tokens } }] }));
    }) as unknown as typeof fetch;
    const s = createGame(1);
    while (s.phase !== "playing") step(s);
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    const d = await teacherPolicy({ baseUrl: "http://teacher.invalid/v1", model: "m", logprobs: true, fetchFn }).decide(enc, undefined, s);
    expect(body.logprobs).toBe(true);
    expect(d.probabilities?.back).toBeCloseTo(0.6 / 0.9, 6);
    if (enc.keys.includes("up")) expect(d.probabilities?.up).toBeCloseTo(0.3 / 0.9, 6);
  });
});

describe("teacher connection reuse", () => {
  const closed = () => new TypeError("fetch failed", { cause: Object.assign(new Error("other side closed"), { code: "UND_ERR_SOCKET" }) });
  const reply = () => new Response(JSON.stringify({ choices: [{ message: { content: '{"move": "back", "reason": "r"}' } }] }));
  const decision = () => {
    const s = createGame(1);
    while (s.phase !== "playing") step(s);
    return { enc: ENCODERS.features.encode(s, findDecisionPoint(s)), s };
  };

  it("asks again when the server closed the idle connection the request went out on", async () => {
    let calls = 0;
    const fetchFn = (async () => {
      calls++;
      if (calls === 1) throw closed();
      return reply();
    }) as unknown as typeof fetch;
    const { enc, s } = decision();
    const d = await teacherPolicy({ baseUrl: "http://teacher.invalid/v1", model: "m", fetchFn }).decide(enc, undefined, s);
    expect(d.choice).toBe("back");
    expect(calls).toBe(2);
  });

  it("asks only once more when the server cannot be reached", async () => {
    let calls = 0;
    const fetchFn = (async () => {
      calls++;
      throw closed();
    }) as unknown as typeof fetch;
    const { enc, s } = decision();
    await expect(teacherPolicy({ baseUrl: "http://teacher.invalid/v1", model: "m", fetchFn }).decide(enc, undefined, s)).rejects.toThrow(/fetch failed/);
    expect(calls).toBe(2);
  });
});
