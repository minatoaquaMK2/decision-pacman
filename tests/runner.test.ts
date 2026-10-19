import { describe, expect, it } from "vitest";
import { greedyPolicy, randomPolicy } from "../src/agent/policies.ts";
import { ENCODERS } from "../src/encoders/index.ts";
import { playGame, type DecisionEvent } from "../src/sim/runner.ts";

describe("playGame alsoEncode", () => {
  it("encodes every decision state again with each extra encoder", async () => {
    const events: DecisionEvent[] = [];
    await playGame({
      seed: 1000,
      policy: greedyPolicy(),
      encoder: ENCODERS.features,
      alsoEncode: [ENCODERS["features-peek5s"]],
      clock: "lockstep",
      maxSeconds: 1,
      onDecision: (ev) => events.push(ev),
    });
    expect(events.length).toBeGreaterThan(5);
    for (const ev of events) {
      const alt = ev.alt!["features-peek5s"];
      expect(alt.keys).toEqual(ev.enc.keys);
      const state = alt.state as Record<string, unknown>;
      expect(state.options).toEqual((ev.enc.state as Record<string, unknown>).options);
      expect(Object.keys(state.lookahead_5s as object).sort()).toEqual([...ev.enc.keys].sort());
    }
  });

  it("does not change the game a player plays", async () => {
    const run = async (alsoEncode?: typeof ENCODERS.features[]) => {
      const states: unknown[] = [];
      const sum = await playGame({
        seed: 1001,
        policy: randomPolicy(1001),
        encoder: ENCODERS.features,
        alsoEncode,
        clock: "lockstep",
        maxSeconds: 10,
        onDecision: (ev) => states.push(ev.enc.state),
      });
      return { sum, states };
    };
    const plain = await run();
    const extra = await run([ENCODERS["features-peek5s"]]);
    expect(extra.states).toEqual(plain.states);
    expect(extra.sum.score).toBe(plain.sum.score);
  });
});
