// Generates labeled training states: a player drives the game, and the teacher
// labels every decision state in the same encoding the student will see.
//
//   npm run gen-data -- --games 20 --seed 1000                      # oracle plays and labels
//   npm run gen-data -- --player random --games 20 --seed 2000      # random play, oracle labels
//   npm run gen-data -- --player teacher --labeler teacher --seconds 30 --seed 1000
//   npm run gen-data -- --player oracle-5s --labeler teacher-peek5s --also-encode features-peek5s --seed 1000
//   npm run gen-data -- --player greedy --labeler teacher-peek5s --label-every 20 --seed 2000   # label 1 state in 20
//
// Players are deterministic for a seed, so a run with the same player and seeds
// visits the same states whatever the labeler. That is how the same states get
// labels from a different teacher. --also-encode stores more encodings of each
// state (under "alt"), for students that read a different input. A teacher
// answers each distinct prompt once per run; repeats are marked label_cached.
// --label-every k asks the labeler on every k-th decision only (the player still
// plays every one), which samples states from whole games at 1/k of the cost.
// Token counts and, when the server reports it, cost go into each row and the report.
//
// The game waits for answers (lockstep clock). Train on seeds 1000 and up,
// validate on 900-999, and keep 100-199 for evaluation only.
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { loadDotEnv, makePolicy, policyEnvFromProcess } from "../src/agent/factory.ts";
import type { Policy, PolicyDecision } from "../src/agent/policies.ts";
import { ENCODERS } from "../src/encoders/index.ts";
import { playGame, percentile, type GameSummary } from "../src/sim/runner.ts";

loadDotEnv();

const { values: args } = parseArgs({
  options: {
    player: { type: "string", default: "oracle-5s" },
    labeler: { type: "string", default: "oracle-5s" },
    encoder: { type: "string", default: "features" },
    "also-encode": { type: "string", default: "" },
    "label-every": { type: "string", default: "1" },
    games: { type: "string", default: "1" },
    seed: { type: "string", default: "1000" },
    seconds: { type: "string", default: "180" },
    speed: { type: "string", default: "1" },
    out: { type: "string", default: "data/teacher" },
  },
});

const EVAL_SEEDS = { min: 100, max: 199 };
export const SCHEMA = "decision-pacman/teacher-v1";

async function main() {
  const env = { ...policyEnvFromProcess(), teacherCache: new Map() };
  const seed0 = Number(args.seed);
  const games = Number(args.games);
  if (seed0 <= EVAL_SEEDS.max && seed0 + games - 1 >= EVAL_SEEDS.min) {
    throw new Error(`seeds ${EVAL_SEEDS.min}-${EVAL_SEEDS.max} are reserved for evaluation`);
  }
  const encoder = ENCODERS[args.encoder!];
  if (!encoder) throw new Error(`unknown encoder ${args.encoder}`);
  const alsoEncode = args["also-encode"]!.split(",").filter(Boolean).map((name) => {
    const e = ENCODERS[name];
    if (!e) throw new Error(`unknown encoder ${name}`);
    return e;
  });

  // Include the encoder and pid so runs started in the same millisecond never share a directory.
  const tag = `${new Date().toISOString().replace(/[:.]/g, "-")}_${args.player}_${args.labeler}_${args.encoder}_s${seed0}_p${process.pid}`.replace(/[^\w.-]/g, "_");
  const dir = join(args.out!, tag);
  mkdirSync(dir, { recursive: true });
  const dataPath = join(dir, "states.jsonl");
  writeFileSync(dataPath, "");

  let labeled = 0;
  let unlabeled = 0;
  let cachedLabels = 0;
  let skipped = 0;
  let inputTokens = 0;
  let outputTokens = 0;
  let reasoningTokens = 0;
  let costUsd = 0;
  let costReported = 0;
  const labelLatency: number[] = [];
  const labelCounts: Record<string, number> = {};
  const errors: string[] = [];
  const summaries: GameSummary[] = [];
  const started = Date.now();
  const labelEvery = Math.max(1, Math.floor(Number(args["label-every"])));

  for (let g = 0; g < games; g++) {
    const seed = seed0 + g;
    const player = makePolicy(args.player!, seed, env);
    const fullLabeler = args.labeler === args.player ? player : makePolicy(args.labeler!, seed, env);
    let asked = 0;
    const labeler: Policy =
      labelEvery === 1 || fullLabeler === player
        ? fullLabeler
        : {
            name: fullLabeler.name,
            requiresEncoder: fullLabeler.requiresEncoder,
            decide: (enc, signal, game) =>
              asked++ % labelEvery === 0
                ? fullLabeler.decide(enc, signal, game)
                : Promise.resolve({ choice: enc.keys[0], latencyMs: 0, skipped: true }),
          };
    const sum = await playGame({
      seed,
      policy: player,
      labeler,
      encoder,
      alsoEncode,
      clock: "lockstep",
      speed: Number(args.speed),
      maxSeconds: Number(args.seconds),
      onDecision: (ev) => {
        const label = ev.label as
          | (PolicyDecision & { reason?: string; cached?: boolean; skipped?: boolean; retries?: number; outputTokens?: number; reasoningTokens?: number; costUsd?: number })
          | null
          | undefined;
        if (label?.skipped) {
          skipped++;
          return;
        }
        if (!label) {
          unlabeled++;
          if (ev.error) errors.push(ev.error);
          return;
        }
        labeled++;
        if (label.cached) cachedLabels++;
        else {
          labelLatency.push(label.latencyMs);
          inputTokens += label.inputTokens ?? 0;
          outputTokens += label.outputTokens ?? 0;
          reasoningTokens += label.reasoningTokens ?? 0;
          if (label.costUsd !== undefined) {
            costUsd += label.costUsd;
            costReported++;
          }
        }
        labelCounts[label.choice] = (labelCounts[label.choice] ?? 0) + 1;
        appendFileSync(
          dataPath,
          JSON.stringify({
            schema: SCHEMA,
            seed,
            tick: ev.askedTick,
            player: player.name,
            labeler: labeler.name,
            encoder: ev.enc.encoder,
            junction: ev.dp.key,
            options: ev.enc.keys,
            state: ev.enc.state,
            instructions: ev.enc.instructions,
            criteria: ev.enc.criteria,
            ...(ev.alt
              ? {
                  alt: Object.fromEntries(
                    Object.entries(ev.alt).map(([name, e]) => [name, { state: e.state, instructions: e.instructions, criteria: e.criteria }]),
                  ),
                }
              : {}),
            label: label.choice,
            values: label.values ?? null,
            ...(label.probabilities ? { label_probs: label.probabilities } : {}),
            reason: label.reason ?? null,
            label_latency_ms: Math.round(label.latencyMs),
            ...(label.cached ? { label_cached: true } : {}),
            ...(!label.cached && label.inputTokens !== undefined
              ? { label_input_tokens: label.inputTokens, label_output_tokens: label.outputTokens ?? null }
              : {}),
            ...(!label.cached && label.reasoningTokens ? { label_reasoning_tokens: label.reasoningTokens } : {}),
            ...(!label.cached && label.retries ? { label_retries: label.retries } : {}),
            ...(!label.cached && label.costUsd !== undefined ? { label_cost_usd: label.costUsd } : {}),
            player_choice: ev.decision?.choice ?? null,
          }) + "\n",
        );
      },
    });
    summaries.push(sum);
    console.log(
      `seed ${seed}: ${sum.secondsPlayed}s played, score ${sum.score}, pellets ${sum.pellets}, deaths ${sum.deaths}, ` +
        `${sum.decisions} decisions, ${sum.failures} failed`,
    );
  }

  const report = {
    schema: SCHEMA,
    player: args.player,
    labeler: args.labeler,
    encoder: args.encoder,
    seeds: [seed0, seed0 + games - 1],
    wallSeconds: Math.round((Date.now() - started) / 1000),
    gameSeconds: summaries.reduce((a, s) => a + s.secondsPlayed, 0),
    labeled,
    unlabeled,
    cachedLabels,
    labelEvery,
    skipped,
    inputTokens,
    outputTokens,
    reasoningTokens,
    costUsd: costReported ? Math.round(costUsd * 1e6) / 1e6 : null,
    labelRequestsPerWallSecond: Math.round(((labeled - cachedLabels) / Math.max(1, (Date.now() - started) / 1000)) * 100) / 100,
    labelsPerWallSecond: Math.round((labeled / Math.max(1, (Date.now() - started) / 1000)) * 100) / 100,
    labelLatencyP50: Math.round(percentile(labelLatency, 0.5)),
    labelLatencyP90: Math.round(percentile(labelLatency, 0.9)),
    labelCounts,
    errors: errors.slice(0, 10),
    games: summaries,
  };
  writeFileSync(join(dir, "report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ ...report, games: undefined, errors: report.errors.length }));
  console.log(`wrote ${dataPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
