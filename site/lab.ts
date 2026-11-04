import { findDecisionPoint, pacTilesPerSecond, type DecisionPoint } from "../src/engine/decision.ts";
import { createGame, step, TPS } from "../src/engine/game.ts";
import type { GameState } from "../src/engine/types.ts";
import { asciiFullEncoder, asciiWindowEncoder } from "../src/encoders/ascii.ts";
import { featuresEncoder } from "../src/encoders/features.ts";
import { optionToInput, type EncodedDecision, type OptionKey } from "../src/encoders/types.ts";
import { peek, rolloutValue, type PeekOutcome } from "../src/agent/oracle.ts";
import { greedyChoice } from "../src/agent/policies.ts";
import { entropyConfidence, optionLetters, renderDecision } from "../src/agent/prompt.ts";

export type ChapterId = "ch1-contract" | "ch2-encoding" | "ch3-distill" | "ch4-ablations" | "ch5-rca";
export type ScenarioId = "opening-fork" | "ghost-pinch" | "power-pellet-chase" | "tunnel-escape";
export type RcaCaseId = "synth-1000" | "bmo-1098" | "installer-10108" | "ironic-137";
export type EncoderId = "features" | "ascii-window" | "ascii-full";
export type HardwareId = "m3-ultra" | "rtx-5090" | "h100-vllm";
export type AblationVariantId =
  | "tev1_08b"
  | "pacman_08b_qwen"
  | "pacman_08b_qwen_h100"
  | "pacman_08b_ds_h100"
  | "pacman_08b_soft_h100"
  | "pacman_08b_mix_h100"
  | "pacman_08b_mlx_1ep";

export interface DiagramNode {
  readonly id: string;
  readonly step: string;
  readonly glyph: string;
  readonly role: "engine" | "encoder" | "model" | "teacher" | "eval";
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
  readonly i18nPrefix: string;
  readonly sourcePath: string;
  readonly codeSnippet: string;
}

export interface DiagramEdge {
  readonly from: string;
  readonly to: string;
  readonly labelKey: string;
}

export interface ChapterSpec {
  readonly id: ChapterId;
  readonly index: number;
  readonly i18nPrefix: string;
  readonly nodes: readonly DiagramNode[];
  readonly edges: readonly DiagramEdge[];
}

export interface ScenarioSnapshot {
  readonly id: ScenarioId;
  readonly seed: number;
  readonly tick: number;
  readonly state: GameState;
  readonly decision: DecisionPoint;
}

export type ScenarioBank = Readonly<Record<ScenarioId, ScenarioSnapshot>>;

export interface SystemOneInspection {
  readonly scenarioId: ScenarioId;
  readonly hardware: HardwareId;
  readonly encoded: EncodedDecision;
  readonly prompt: string;
  readonly optionMap: readonly {
    readonly letter: string;
    readonly key: OptionKey;
    readonly probability: number;
  }[];
  readonly chosenKey: OptionKey;
  readonly chosenLetter: string;
  readonly confidence: number;
  readonly prefillMs: number;
  readonly decodePerTokenMs: number;
  readonly oneTokenTotalMs: number;
  readonly chatSevenTokenMs: number;
}

export interface EncoderPayloadSummary {
  readonly id: EncoderId;
  readonly text: string;
  readonly chars: number;
  readonly estimatedTokens: number;
  readonly observedConfidenceBand: string;
}

export interface EncoderHorizonInspection {
  readonly scenarioId: ScenarioId;
  readonly payloads: Readonly<Record<EncoderId, EncoderPayloadSummary>>;
  readonly pacSpeedTilesPerSec: number;
  readonly latencyMs: number;
  readonly tilesTraveledDuringCall: number;
  readonly junctionDistanceTiles: number;
  readonly arrivalMarginTiles: number;
  readonly isStale: boolean;
}

export interface OptionRolloutSummary {
  readonly key: OptionKey;
  readonly letter: string;
  readonly studentFacts: Record<string, unknown>;
  readonly teacherPeek: PeekOutcome;
  readonly rolloutScore: number;
}

export interface DistillationInspection {
  readonly scenarioId: ScenarioId;
  readonly horizonSeconds: number;
  readonly options: readonly OptionRolloutSummary[];
  readonly studentGreedyMove: OptionKey;
  readonly teacherLookaheadMove: OptionKey;
  readonly informationGapExposed: boolean;
}

export interface LogprobMaskInspection {
  readonly mode: "raw" | "processed";
  readonly validLetters: readonly string[];
  readonly rawTopK: readonly { readonly token: string; readonly prob: number; readonly isOption: boolean }[];
  readonly retainedOptions: Readonly<Record<string, number>>;
  readonly droppedValidLetters: readonly string[];
}

export interface AblationRow {
  readonly id: AblationVariantId;
  readonly teacher: string;
  readonly target: "hard" | "soft" | "mixed" | "n/a";
  readonly hardware: string;
  readonly realtime: number;
  readonly lockstep: number;
  readonly se: number;
  readonly jevbench: number;
}

export interface AblationComparison {
  readonly candidate: AblationRow;
  readonly baseline: AblationRow;
  readonly clock: "realtime" | "lockstep";
  readonly deltaPellets: number;
  readonly combinedSe: number;
  readonly zScore: number;
  readonly withinNoise: boolean;
  readonly jevbenchDelta: number;
}

export interface RcaCaseSnapshot {
  readonly id: RcaCaseId;
  readonly source: string;
  readonly split: "synthetic" | "real";
  readonly humanLabel: string;
  readonly studentChoice: string;
  readonly studentLatencyMs: number;
  readonly deepseekZeroShot: string;
  readonly teacherWithComments: string;
  readonly topProbs: readonly { readonly option: string; readonly prob: number }[];
  readonly state: {
    readonly stage: string;
    readonly bmh_state: string;
    readonly events: readonly string[];
    readonly logs: readonly string[];
    readonly checks: readonly string[];
    readonly environment: {
      readonly hardware: string;
      readonly ocp_version: string;
      readonly network: string;
    };
  };
  readonly diagnosisKey: string;
}

export interface RcaCaseInspection {
  readonly snapshot: RcaCaseSnapshot;
  readonly studentCorrect: boolean;
  readonly deepseekCorrect: boolean;
  readonly teacherAuditCorrect: boolean;
  readonly formattedStateJson: string;
}

export const CHAPTER_SPECS: readonly ChapterSpec[] = [
  {
    id: "ch1-contract",
    index: 1,
    i18nPrefix: "lab.ch1",
    nodes: [
      {
        id: "ch1-n1",
        step: "01",
        glyph: "dp",
        role: "engine",
        x: 20,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch1.n1",
        sourcePath: "src/engine/decision.ts",
        codeSnippet:
          "const dp = findDecisionPoint(s, lookahead);\n" +
          "const keys = optionKeys(dp); // e.g. ['up', 'left', 'back']",
      },
      {
        id: "ch1-n2",
        step: "02",
        glyph: "{}",
        role: "encoder",
        x: 272,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch1.n2",
        sourcePath: "src/agent/prompt.ts",
        codeSnippet:
          "const letters = optionLetters(keys.length); // ['A', 'B', 'C']\n" +
          "const prompt = renderPrompt(enc.state, enc.instructions, enc.criteria);",
      },
      {
        id: "ch1-n3",
        step: "03",
        glyph: "P(A)",
        role: "model",
        x: 524,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch1.n3",
        sourcePath: "src/agent/client.ts",
        codeSnippet:
          "POST /v1/systemone\n" +
          "-> scores logits over option letters A..D in 1 forward pass\n" +
          "-> confidence = 1 - H(p) / ln(N)",
      },
      {
        id: "ch1-n4",
        step: "04",
        glyph: "30Hz",
        role: "engine",
        x: 776,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch1.n4",
        sourcePath: "src/agent/loop.ts",
        codeSnippet:
          "if (isStillAhead(state, dp, lookahead)) {\n" +
          "  latestInput = optionToInput(decision.choice, dp);\n" +
          "}",
      },
    ],
    edges: [
      { from: "ch1-n1", to: "ch1-n2", labelKey: "lab.ch1.e1" },
      { from: "ch1-n2", to: "ch1-n3", labelKey: "lab.ch1.e2" },
      { from: "ch1-n3", to: "ch1-n4", labelKey: "lab.ch1.e3" },
    ],
  },
  {
    id: "ch2-encoding",
    index: 2,
    i18nPrefix: "lab.ch2",
    nodes: [
      {
        id: "ch2-n1",
        step: "01",
        glyph: "BFS",
        role: "engine",
        x: 20,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch2.n1",
        sourcePath: "src/engine/decision.ts",
        codeSnippet:
          "export function distancesThrough(x: number, y: number, first: Dir): Int16Array {\n" +
          "  // exact maze distances through each exit\n" +
          "}",
      },
      {
        id: "ch2-n2",
        step: "02",
        glyph: "JSON",
        role: "encoder",
        x: 272,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch2.n2",
        sourcePath: "src/encoders/features.ts",
        codeSnippet:
          "const facts: OptionFacts = {\n" +
          "  ghost: round(ghost), ghost_coming,\n" +
          "  pellets, food: round(food), power: round(power)\n" +
          "};",
      },
      {
        id: "ch2-n3",
        step: "03",
        glyph: "v*t",
        role: "engine",
        x: 524,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch2.n3",
        sourcePath: "src/agent/loop.ts",
        codeSnippet:
          "const budgetTiles = pacTilesPerSecond(s) * (latencyMs / 1000);\n" +
          "// 7.58 tiles/s * 0.055s = 0.42 tiles per call",
      },
      {
        id: "ch2-n4",
        step: "04",
        glyph: "turn",
        role: "engine",
        x: 776,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch2.n4",
        sourcePath: "src/engine/game.ts",
        codeSnippet:
          "p.turns.push({ x: dp.x, y: dp.y, dir: choice });\n" +
          "// buffered turn consumed cleanly on tile center arrival",
      },
    ],
    edges: [
      { from: "ch2-n1", to: "ch2-n2", labelKey: "lab.ch2.e1" },
      { from: "ch2-n2", to: "ch2-n3", labelKey: "lab.ch2.e2" },
      { from: "ch2-n3", to: "ch2-n4", labelKey: "lab.ch2.e3" },
    ],
  },
  {
    id: "ch3-distill",
    index: 3,
    i18nPrefix: "lab.ch3",
    nodes: [
      {
        id: "ch3-n1",
        step: "01",
        glyph: "5s",
        role: "engine",
        x: 20,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch3.n1",
        sourcePath: "src/agent/oracle.ts",
        codeSnippet:
          "export function peek(s0: GameState, enc: EncodedDecision, key: string, ticks = 150): PeekOutcome {\n" +
          "  // 5s rollout: dies_after_s, pellets, power_pellets, points\n" +
          "}",
      },
      {
        id: "ch3-n2",
        step: "02",
        glyph: "27B",
        role: "teacher",
        x: 272,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch3.n2",
        sourcePath: "src/agent/teacher.ts",
        codeSnippet:
          "teacher-peek5s (Qwen3.5-27B)\n" +
          "-> sees features + 5s rollout outcomes at labeling time\n" +
          "-> labels 54,193 junction states",
      },
      {
        id: "ch3-n3",
        step: "03",
        glyph: "LoRA",
        role: "model",
        x: 524,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch3.n3",
        sourcePath: "training/train.py",
        codeSnippet:
          "LoRA r=32, alpha=64 on Qwen3.5-0.8B-Base\n" +
          "student input = features ONLY (no rollout peek at inference!)",
      },
      {
        id: "ch3-n4",
        step: "04",
        glyph: "Q8_0",
        role: "model",
        x: 776,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch3.n4",
        sourcePath: "training/export_gguf.sh",
        codeSnippet:
          "pacman-0.8b-qwen-Q8_0.gguf (795 MB)\n" +
          "-> 425 pellets realtime (55 ms/call)\n" +
          "-> runs offline on iPhone (~505 ms/step)",
      },
    ],
    edges: [
      { from: "ch3-n1", to: "ch3-n2", labelKey: "lab.ch3.e1" },
      { from: "ch3-n2", to: "ch3-n3", labelKey: "lab.ch3.e2" },
      { from: "ch3-n3", to: "ch3-n4", labelKey: "lab.ch3.e3" },
    ],
  },
  {
    id: "ch4-ablations",
    index: 4,
    i18nPrefix: "lab.ch4",
    nodes: [
      {
        id: "ch4-n1",
        step: "01",
        glyph: "mask",
        role: "teacher",
        x: 20,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch4.n1",
        sourcePath: "scripts/systemone_vllm.ts",
        codeSnippet:
          "vLLM --logprobs-mode processed_logprobs\n" +
          "// applies guided_choice mask BEFORE top-k truncation",
      },
      {
        id: "ch4-n2",
        step: "02",
        glyph: "KL",
        role: "model",
        x: 272,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch4.n2",
        sourcePath: "training/build_sft.py",
        codeSnippet:
          "Hard CE (425) vs Soft KL (409) vs 50/50 Mix (481)\n" +
          "// teacher averages 0.996 top-1 probability -> near one-hot",
      },
      {
        id: "ch4-n3",
        step: "03",
        glyph: "SE",
        role: "eval",
        x: 524,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch4.n3",
        sourcePath: "docs/model_evaluation.md",
        codeSnippet:
          "10-game SE = ~30 pellets (min significant gap ~85)\n" +
          "All 6 teacher/hardware/target variants land in 409-481",
      },
      {
        id: "ch4-n4",
        step: "04",
        glyph: "0.53",
        role: "eval",
        x: 776,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch4.n4",
        sourcePath: "docs/results.md",
        codeSnippet:
          "Pac-Man: 69 -> 425 pellets (6.2x)\n" +
          "JevBench (194 general tasks): 0.66 -> 0.53",
      },
    ],
    edges: [
      { from: "ch4-n1", to: "ch4-n2", labelKey: "lab.ch4.e1" },
      { from: "ch4-n2", to: "ch4-n3", labelKey: "lab.ch4.e2" },
      { from: "ch4-n3", to: "ch4-n4", labelKey: "lab.ch4.e3" },
    ],
  },
  {
    id: "ch5-rca",
    index: 5,
    i18nPrefix: "lab.ch5",
    nodes: [
      {
        id: "ch5-n1",
        step: "01",
        glyph: "23",
        role: "encoder",
        x: 20,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch5.n1",
        sourcePath: "src/agent/prompt.ts",
        codeSnippet:
          "schema: 'decision-pacman/teacher-v1', encoder: 'rca-facts'\n" +
          "options: [23 bare-metal root causes -> letters A..W]",
      },
      {
        id: "ch5-n2",
        step: "02",
        glyph: "3m",
        role: "teacher",
        x: 272,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch5.n2",
        sourcePath: "training/train.py",
        codeSnippet:
          "1,128 Qwen3.8-27B doc summaries -> LoRA on Qwen3.5-0.8B\n" +
          "141 steps (3 min on 1 H100): val acc 21.8% -> 90.2%",
      },
      {
        id: "ch5-n3",
        step: "03",
        glyph: "94%",
        role: "model",
        x: 524,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch5.n3",
        sourcePath: "training/score_ollama.py",
        codeSnippet:
          "333 held-out synthetic doc cases:\n" +
          "rca-0.8b: 94.0% (76 ms, 1 H100) vs 671B MoE: 81.7% (145 ms, 8 H100s)",
      },
      {
        id: "ch5-n4",
        step: "04",
        glyph: "23%",
        role: "eval",
        x: 776,
        y: 55,
        width: 205,
        height: 114,
        i18nPrefix: "lab.ch5.n4",
        sourcePath: "scripts/systemone_vllm.ts",
        codeSnippet:
          "48 settled GitHub issues (34 product_bug, 14 config/env):\n" +
          "rca-0.8b: 22.9% (11/48) vs 671B MoE zero-shot: 41.7% (20/48)",
      },
    ],
    edges: [
      { from: "ch5-n1", to: "ch5-n2", labelKey: "lab.ch5.e1" },
      { from: "ch5-n2", to: "ch5-n3", labelKey: "lab.ch5.e2" },
      { from: "ch5-n3", to: "ch5-n4", labelKey: "lab.ch5.e3" },
    ],
  },
];

const HARDWARE_TIMINGS: Readonly<Record<HardwareId, { prefillMs: number; decodePerTokenMs: number }>> = {
  "m3-ultra": { prefillMs: 41, decodePerTokenMs: 5.5 },
  "rtx-5090": { prefillMs: 17, decodePerTokenMs: 2.1 },
  "h100-vllm": { prefillMs: 12, decodePerTokenMs: 2.2 },
};

let cachedSeed: number | null = null;
let cachedBank: ScenarioBank | null = null;

export function buildScenarioBank(seed = 100): ScenarioBank {
  if (cachedSeed === seed && cachedBank !== null) return cachedBank;

  const s = createGame(seed);
  while (s.phase !== "playing") step(s);

  const candidates: { tick: number; state: GameState; decision: DecisionPoint }[] = [];
  let lastKey = "";

  for (let t = 0; t < 750 && s.phase === "playing"; t++) {
    const dp = findDecisionPoint(s);
    if (dp.exits.length > 1 && dp.key !== lastKey && dp.distance <= 3.2) {
      candidates.push({
        tick: s.tick,
        state: structuredClone(s),
        decision: structuredClone(dp),
      });
      const enc = featuresEncoder.encode(s, dp);
      const choice = greedyChoice(enc);
      step(s, optionToInput(choice, dp) ?? {});
      lastKey = dp.key;
      continue;
    }
    step(s);
  }

  const usedIndices = new Set<number>();
  const usedCoords = new Set<string>();
  const pickCandidate = (predicate: (c: { tick: number; state: GameState; decision: DecisionPoint }) => boolean): {
    tick: number;
    state: GameState;
    decision: DecisionPoint;
  } => {
    for (let i = 0; i < candidates.length; i++) {
      const coord = `${candidates[i].decision.x},${candidates[i].decision.y}`;
      if (!usedIndices.has(i) && !usedCoords.has(coord) && predicate(candidates[i])) {
        usedIndices.add(i);
        usedCoords.add(coord);
        return candidates[i];
      }
    }
    for (let i = 0; i < candidates.length; i++) {
      const coord = `${candidates[i].decision.x},${candidates[i].decision.y}`;
      if (!usedIndices.has(i) && !usedCoords.has(coord)) {
        usedIndices.add(i);
        usedCoords.add(coord);
        return candidates[i];
      }
    }
    for (let i = 0; i < candidates.length; i++) {
      if (!usedIndices.has(i)) {
        usedIndices.add(i);
        return candidates[i];
      }
    }
    return candidates[0];
  };

  const cOpening = pickCandidate(() => true);

  const cGhostPinch = pickCandidate((c) => {
    const enc = featuresEncoder.encode(c.state, c.decision);
    const opts = (enc.state as { options: Record<string, { ghost: number | null }> }).options;
    return Object.values(opts).some((o) => o.ghost !== null && o.ghost <= 10);
  });

  const cPowerChase = pickCandidate((c) => {
    const enc = featuresEncoder.encode(c.state, c.decision);
    const opts = (enc.state as { options: Record<string, { power?: number | null }> }).options;
    return c.state.frightTicks > 0 || Object.values(opts).some((o) => o.power !== null && o.power !== undefined && o.power <= 14);
  });

  const cTunnelEscape = pickCandidate((c) => {
    const enc = featuresEncoder.encode(c.state, c.decision);
    const greedy = greedyChoice(enc);
    const horizonTicks = 5 * TPS;
    let bestKey = enc.keys[0];
    let bestVal = -Infinity;
    for (const k of enc.keys) {
      const v = rolloutValue(c.state, enc, k, horizonTicks, 0);
      if (v > bestVal) {
        bestVal = v;
        bestKey = k;
      }
    }
    return bestKey !== greedy;
  });

  const bank: ScenarioBank = {
    "opening-fork": {
      id: "opening-fork",
      seed,
      tick: cOpening.tick,
      state: structuredClone(cOpening.state),
      decision: structuredClone(cOpening.decision),
    },
    "ghost-pinch": {
      id: "ghost-pinch",
      seed,
      tick: cGhostPinch.tick,
      state: structuredClone(cGhostPinch.state),
      decision: structuredClone(cGhostPinch.decision),
    },
    "power-pellet-chase": {
      id: "power-pellet-chase",
      seed,
      tick: cPowerChase.tick,
      state: structuredClone(cPowerChase.state),
      decision: structuredClone(cPowerChase.decision),
    },
    "tunnel-escape": {
      id: "tunnel-escape",
      seed,
      tick: cTunnelEscape.tick,
      state: structuredClone(cTunnelEscape.state),
      decision: structuredClone(cTunnelEscape.decision),
    },
  };

  cachedSeed = seed;
  cachedBank = bank;
  return bank;
}

export function inspectSystemOneContract(snapshot: ScenarioSnapshot, hardware: HardwareId = "m3-ultra"): SystemOneInspection {
  const encoded = featuresEncoder.encode(snapshot.state, snapshot.decision);
  const prompt = renderDecision(encoded);
  const letters = optionLetters(encoded.keys.length);
  const horizonTicks = 5 * TPS;

  const scores = encoded.keys.map((k) => rolloutValue(snapshot.state, encoded, k, horizonTicks, 0));
  const maxScore = Math.max(...scores);
  const temp = 35;
  const exps = scores.map((s) => Math.exp(Math.max(-20, (s - maxScore) / temp)));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => Math.round((e / sumExp) * 1000) / 1000);

  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIdx]) bestIdx = i;
  }

  const optionMap = encoded.keys.map((key, i) => ({
    letter: letters[i],
    key,
    probability: probs[i],
  }));

  const confidence = Math.round(entropyConfidence(probs) * 1000) / 1000;
  const timing = HARDWARE_TIMINGS[hardware];
  const oneTokenTotalMs = Math.round(timing.prefillMs);
  const chatSevenTokenMs = Math.round(timing.prefillMs + 6 * timing.decodePerTokenMs);

  return {
    scenarioId: snapshot.id,
    hardware,
    encoded,
    prompt,
    optionMap,
    chosenKey: encoded.keys[bestIdx],
    chosenLetter: letters[bestIdx],
    confidence,
    prefillMs: timing.prefillMs,
    decodePerTokenMs: timing.decodePerTokenMs,
    oneTokenTotalMs,
    chatSevenTokenMs,
  };
}

function formatEncodedText(enc: EncodedDecision): string {
  return typeof enc.state === "string" ? enc.state : JSON.stringify(enc.state, null, 2);
}

export function inspectEncodersAndHorizon(snapshot: ScenarioSnapshot, latencyMs: number): EncoderHorizonInspection {
  const feat = featuresEncoder.encode(snapshot.state, snapshot.decision);
  const win = asciiWindowEncoder.encode(snapshot.state, snapshot.decision);
  const full = asciiFullEncoder.encode(snapshot.state, snapshot.decision);

  const toSummary = (id: EncoderId, enc: EncodedDecision, band: string): EncoderPayloadSummary => {
    const text = formatEncodedText(enc);
    const fullPrompt = renderDecision(enc);
    return {
      id,
      text,
      chars: text.length,
      estimatedTokens: Math.round(fullPrompt.length / 3.6),
      observedConfidenceBand: band,
    };
  };

  const payloads: Record<EncoderId, EncoderPayloadSummary> = {
    features: toSummary("features", feat, "0.60 - 0.90"),
    "ascii-window": toSummary("ascii-window", win, "0.02 - 0.14"),
    "ascii-full": toSummary("ascii-full", full, "0.01 - 0.08"),
  };

  const pacSpeedTilesPerSec = Math.round(pacTilesPerSecond(snapshot.state) * 100) / 100;
  const tilesTraveledDuringCall = Math.round(pacSpeedTilesPerSec * (latencyMs / 1000) * 100) / 100;
  const junctionDistanceTiles = Math.round(snapshot.decision.distance * 100) / 100;
  const arrivalMarginTiles = Math.round((junctionDistanceTiles - tilesTraveledDuringCall) * 100) / 100;
  const isStale = tilesTraveledDuringCall > junctionDistanceTiles;

  return {
    scenarioId: snapshot.id,
    payloads,
    pacSpeedTilesPerSec,
    latencyMs,
    tilesTraveledDuringCall,
    junctionDistanceTiles,
    arrivalMarginTiles,
    isStale,
  };
}

export function inspectDistillationGap(snapshot: ScenarioSnapshot, horizonSeconds = 5): DistillationInspection {
  const encoded = featuresEncoder.encode(snapshot.state, snapshot.decision);
  const letters = optionLetters(encoded.keys.length);
  const horizonTicks = Math.max(1, Math.round(horizonSeconds * TPS));
  const rawOptions = (encoded.state as { options: Record<string, Record<string, unknown>> }).options;

  let bestRolloutKey = encoded.keys[0];
  let bestRolloutScore = -Infinity;

  const options: OptionRolloutSummary[] = encoded.keys.map((key, idx) => {
    const teacherPeek = peek(snapshot.state, encoded, key, horizonTicks);
    const rolloutScore = Math.round(rolloutValue(snapshot.state, encoded, key, horizonTicks, 0) * 10) / 10;
    if (rolloutScore > bestRolloutScore) {
      bestRolloutScore = rolloutScore;
      bestRolloutKey = key;
    }
    return {
      key,
      letter: letters[idx],
      studentFacts: rawOptions[key] ?? {},
      teacherPeek,
      rolloutScore,
    };
  });

  const studentGreedyMove = greedyChoice(encoded) as OptionKey;
  const teacherLookaheadMove = bestRolloutKey;
  const anyDeathTrap = options.some((o) => o.teacherPeek.dies_after_s !== null);

  return {
    scenarioId: snapshot.id,
    horizonSeconds,
    options,
    studentGreedyMove,
    teacherLookaheadMove,
    informationGapExposed: studentGreedyMove !== teacherLookaheadMove || anyDeathTrap,
  };
}

export function inspectProcessedLogprobs(
  vocabProbs: Record<string, number>,
  validOptionCount: number,
  topK: number,
  mode: "raw" | "processed" = "raw",
): LogprobMaskInspection {
  const validLetters = optionLetters(validOptionCount);
  const validSet = new Set(validLetters);

  const sortedAll = Object.entries(vocabProbs)
    .map(([token, prob]) => ({ token, prob, isOption: validSet.has(token) }))
    .sort((a, b) => b.prob - a.prob);

  const pool =
    mode === "processed"
      ? sortedAll.filter((entry) => entry.isOption)
      : sortedAll;

  const rawTopK = pool.slice(0, topK);
  const retainedOptions: Record<string, number> = {};
  for (const item of rawTopK) {
    if (item.isOption) retainedOptions[item.token] = item.prob;
  }

  const droppedValidLetters = validLetters.filter((letter) => !(letter in retainedOptions));

  return {
    mode,
    validLetters,
    rawTopK,
    retainedOptions,
    droppedValidLetters,
  };
}

export function compareAblationRun(
  rows: readonly AblationRow[],
  candidateId: AblationVariantId,
  baselineId: AblationVariantId = "pacman_08b_qwen",
  clock: "realtime" | "lockstep" = "realtime",
): AblationComparison {
  const candidate = rows.find((r) => r.id === candidateId) ?? rows[0];
  const baseline = rows.find((r) => r.id === baselineId) ?? rows[1] ?? rows[0];

  const deltaPellets = candidate[clock] - baseline[clock];
  const combinedSe = Math.round(Math.hypot(candidate.se, baseline.se) * 10) / 10;
  const zScore = combinedSe > 0 ? Math.round((deltaPellets / combinedSe) * 100) / 100 : 0;
  const withinNoise = Math.abs(zScore) < 1.96;
  const jevbenchDelta = Math.round((candidate.jevbench - baseline.jevbench) * 100) / 100;

  return {
    candidate,
    baseline,
    clock,
    deltaPellets,
    combinedSe,
    zScore,
    withinNoise,
    jevbenchDelta,
  };
}

export const RCA_CASES: Readonly<Record<RcaCaseId, RcaCaseSnapshot>> = {
  "synth-1000": {
    id: "synth-1000",
    source: "eval_synth/gen_s1000#0",
    split: "synthetic",
    humanLabel: "bmc_credentials",
    studentChoice: "bmc_credentials",
    studentLatencyMs: 76,
    deepseekZeroShot: "bmc_credentials",
    teacherWithComments: "bmc_credentials",
    topProbs: [
      { option: "bmc_credentials", prob: 0.891 },
      { option: "ramdisk_callback_blocked", prob: 0.011 },
      { option: "image_download_failed", prob: 0.007 },
    ],
    state: {
      stage: "cluster install (baremetal deployment)",
      bmh_state: "registering",
      events: [
        "RegistrationError: failed to get power state for node 7f3c9d2e-1a4b. HTTP GET https://192.0.2.10:443/redfish/v1/Systems/1 returned code 401.",
        "ProvisioningFailed: unable to verify current power status of host, aborting provisioning",
      ],
      logs: [
        "metal3-baremetal-controller: Failed to get power state for node 3e8f6c2a-9b1d. HTTP GET https://192.0.2.10:443/redfish/v1/Systems/1 returned code 401.",
        "ironic-conductor: [Errno 401] Unauthorized for BMC access at https://192.0.2.10:443",
      ],
      checks: [
        "curl -k -u admin:**** https://192.0.2.10/redfish/v1/Systems/1 returned HTTP 401",
        "ipmitool -I lanplus -H 192.0.2.10 -U admin -P **** power status returned 'Unable to establish IPMI v2 / RMCP+ session'",
        "ping 192.0.2.10 -c 3 succeeded, HTTPS port 443 responded with 401",
      ],
      environment: {
        hardware: "Cisco UCS C240 M6 CIMC",
        ocp_version: "4.17",
        network: "BMC 192.0.2.10/24, prov 198.51.100.0/24",
      },
    },
    diagnosisKey: "lab.ch5.cases.synth_1000_diag",
  },
  "bmo-1098": {
    id: "bmo-1098",
    source: "metal3-io/baremetal-operator#1098",
    split: "real",
    humanLabel: "bmc_driver_mismatch",
    studentChoice: "bmc_driver_mismatch",
    studentLatencyMs: 74,
    deepseekZeroShot: "bmc_driver_mismatch",
    teacherWithComments: "product_bug",
    topProbs: [
      { option: "bmc_driver_mismatch", prob: 0.616 },
      { option: "product_bug", prob: 0.145 },
      { option: "install_config_error", prob: 0.046 },
    ],
    state: {
      stage: "registration",
      bmh_state: "error",
      events: [
        "BareMetalHost configured with BMC address using http scheme",
        "BareMetalHost status shows error count 12 and registration error",
      ],
      logs: [
        "Failed to get power state: Redfish connection failed: Unable to connect to https://203.0.113.71:8000/redfish/v1/ (SSLError: [SSL: WRONG_VERSION_NUMBER])",
      ],
      checks: [
        "BMC address in spec is redfish+http://203.0.113.71:8000/redfish/v1/Systems/...",
        "Disable certificate verification is set to true",
      ],
      environment: {
        hardware: "emulated VM with sushy BMC",
        ocp_version: "ACM 2.4.2",
        network: "203.0.113.71:8000",
      },
    },
    diagnosisKey: "lab.ch5.cases.bmo_1098_diag",
  },
  "installer-10108": {
    id: "installer-10108",
    source: "openshift/installer#10108",
    split: "real",
    humanLabel: "product_bug",
    studentChoice: "api_ingress_vip_dns",
    studentLatencyMs: 74,
    deepseekZeroShot: "product_bug",
    teacherWithComments: "product_bug",
    topProbs: [
      { option: "api_ingress_vip_dns", prob: 0.869 },
      { option: "ntp_clock_skew", prob: 0.01 },
      { option: "product_bug", prob: 0.006 },
    ],
    state: {
      stage: "deployment",
      bmh_state: "unknown",
      events: [
        "User set both apiVIPs and ingressVIPs to 192.0.2.100 with loadBalancer: UserManaged in install-config.yaml",
        "ISO generation succeeded, but assisted-service rejected duplicate VIP when node booted",
        "Inspection of generated AgentClusterInstall manifest on node showed loadBalancer stanza was dropped",
      ],
      logs: [
        "FATAL failed to fetch Agent Installer ISO: invalid install-config configuration: platform.baremetal.apiVIPs: Invalid value: \"192.0.2.100\": VIP for API must not be one of the Ingress VIPs",
      ],
      checks: [
        "Validation passes when loadBalancer type UserManaged is set",
        "AgentClusterInstall manifest on node lacks loadBalancer type UserManaged stanza",
      ],
      environment: {
        hardware: "3 master, 4 worker replicas",
        ocp_version: "4.20.1",
        network: "apiVIPs 192.0.2.100, ingressVIPs 192.0.2.100",
      },
    },
    diagnosisKey: "lab.ch5.cases.installer_10108_diag",
  },
  "ironic-137": {
    id: "ironic-137",
    source: "metal3-io/ironic-image#137",
    split: "real",
    humanLabel: "boot_mode_mismatch",
    studentChoice: "product_bug",
    studentLatencyMs: 73,
    deepseekZeroShot: "product_bug",
    teacherWithComments: "product_bug",
    topProbs: [
      { option: "product_bug", prob: 0.802 },
      { option: "install_config_error", prob: 0.035 },
      { option: "boot_mode_mismatch", prob: 0.021 },
    ],
    state: {
      stage: "enrollment",
      bmh_state: "provisioning",
      events: [
        "Error creating ISO due to missing isolinux.bin file",
        "Logged into ironic-conductor pod and confirmed /usr/lib/syslinux/isolinux.bin is absent",
        "Installed syslinux package and found file at /usr/share/syslinux/isolinux.bin",
      ],
      logs: [
        "Error: couldn't create ISO because isolinux.bin file does not exist",
      ],
      checks: [
        "Verified absence of /usr/lib/syslinux/isolinux.bin in ironic-conductor",
        "Confirmed syslinux package installs file to /usr/share/syslinux/isolinux.bin",
      ],
      environment: {
        hardware: "bare-metal",
        ocp_version: "unknown",
        network: "virtualmedia",
      },
    },
    diagnosisKey: "lab.ch5.cases.ironic_137_diag",
  },
};

export function inspectRcaCase(caseId: RcaCaseId): RcaCaseInspection {
  const snapshot = RCA_CASES[caseId] ?? RCA_CASES["synth-1000"];
  return {
    snapshot,
    studentCorrect: snapshot.studentChoice === snapshot.humanLabel,
    deepseekCorrect: snapshot.deepseekZeroShot === snapshot.humanLabel,
    teacherAuditCorrect: snapshot.teacherWithComments === snapshot.humanLabel,
    formattedStateJson: JSON.stringify(snapshot.state, null, 2),
  };
}
