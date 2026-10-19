import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import { renderPrompt } from "../src/agent/prompt.ts";
import { systemOneVllmServer } from "../scripts/systemone_vllm.ts";

const servers: Server[] = [];

async function listen(server: Server): Promise<string> {
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}

/** A vLLM stand-in: letters A..Z are tokens 100..125, and completions answer with fixed letter log-probabilities. */
async function fakeVllm(models: string[], top: Record<string, number>) {
  const completions: Record<string, unknown>[] = [];
  const url = await listen(
    createServer(async (req, res) => {
      let raw = "";
      for await (const chunk of req) raw += chunk;
      const body = JSON.parse(raw);
      res.setHeader("Content-Type", "application/json");
      if (!models.includes(body.model)) {
        res.statusCode = 404;
        return res.end(JSON.stringify({ error: { message: `The model \`${body.model}\` does not exist.` } }));
      }
      if (req.url === "/tokenize") return res.end(JSON.stringify({ tokens: [100 + body.prompt.charCodeAt(0) - 65] }));
      completions.push(body);
      res.end(JSON.stringify({ choices: [{ text: "B", logprobs: { top_logprobs: [top] } }], usage: { prompt_tokens: 321 } }));
    }),
  );
  return { url, completions };
}

async function ask(url: string, body: object) {
  const res = await fetch(`${url}/v1/systemone`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return { status: res.status, body: await res.json() };
}

const question = { type: "choice", instructions: "Which way?", criteria: { up: null, left: null, back: "turn around now" } };
const state = { lives: 3, options: { up: { ghost: null } } };

afterEach(async () => {
  await Promise.all(servers.splice(0).map((s) => new Promise((resolve) => s.close(resolve))));
});

describe("systemone_vllm", () => {
  it("answers like Ollama: softmax over the option letters, keyed by option", async () => {
    const vllm = await fakeVllm(["pacman"], { B: Math.log(0.6), C: Math.log(0.3), A: Math.log(0.05) });
    const adapter = await listen(systemOneVllmServer(vllm.url));
    const { status, body } = await ask(adapter, { model: "pacman", keep_alive: -1, state, questions: { move: question } });

    expect(status).toBe(200);
    expect(body.answers.move.choice).toBe("left");
    const p = body.answers.move.probabilities;
    expect(p.up + p.left + p.back).toBeCloseTo(1, 9);
    expect(p.left).toBeCloseTo(0.6 / 0.95, 9);
    expect(p.up).toBeCloseTo(0.05 / 0.95, 9);
    expect(body.answers.move.confidence).toBeGreaterThan(0);
    expect(body.answers.move.confidence).toBeLessThan(1);
    expect(body.usage.input_tokens).toBe(321);
  });

  it("asks vLLM for one token over exactly the option letters, on the prompt Ollama scores", async () => {
    const vllm = await fakeVllm(["pacman"], { A: -1, B: -1, C: -1 });
    const adapter = await listen(systemOneVllmServer(vllm.url));
    await ask(adapter, { model: "pacman", state, questions: { move: question } });

    expect(vllm.completions).toEqual([
      {
        model: "pacman",
        prompt: renderPrompt(state, question.instructions, question.criteria, "move"),
        max_tokens: 1,
        temperature: 0,
        logprobs: 3,
        allowed_token_ids: [100, 101, 102],
      },
    ]);
  });

  it("reports a model vLLM does not serve as 404, which the game client shows as a missing model", async () => {
    const vllm = await fakeVllm(["pacman"], {});
    const adapter = await listen(systemOneVllmServer(vllm.url));
    const { status, body } = await ask(adapter, { model: "other", state, questions: { move: question } });

    expect(status).toBe(404);
    expect(body.error).toMatch(/does not exist/);
  });

  it("fails instead of guessing when vLLM leaves an option letter out", async () => {
    const vllm = await fakeVllm(["pacman"], { A: -0.7, C: -2.0, D: -2.1 });
    const adapter = await listen(systemOneVllmServer(vllm.url));
    const { status, body } = await ask(adapter, { model: "pacman", state, questions: { move: question } });

    expect(status).toBe(502);
    expect(body.error).toMatch(/letters B .*processed_logprobs/);
  });

  it("rejects a question that is not a choice", async () => {
    const vllm = await fakeVllm(["pacman"], {});
    const adapter = await listen(systemOneVllmServer(vllm.url));
    const { status } = await ask(adapter, { model: "pacman", state, questions: { move: { type: "text" } } });

    expect(status).toBe(400);
  });
});
