import { existsSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { fill, flattenKeys, lookup, pickLang } from "../site/i18n.ts";
import { rankPlayer } from "../site/ranking.ts";
import { compactState, encodeAhead, holdForPlayer } from "../site/eyes.ts";
import {
  CHAPTER_SPECS,
  RCA_CASES,
  buildScenarioBank,
  compareAblationRun,
  inspectDistillationGap,
  inspectEncodersAndHorizon,
  inspectProcessedLogprobs,
  inspectRcaCase,
  inspectSystemOneContract,
  type AblationRow,
  type RcaCaseId,
} from "../site/lab.ts";
import { findDecisionPoint } from "../src/engine/decision.ts";
import { createGame, step } from "../src/engine/game.ts";
import data from "../site/data.json";
import { accuracyText, evaluation, evaluationMarkup, summarizeSpend, type UsageRow } from "../site/evaluation.ts";
import { scenarioCost, workflowCostMarkup, workflowMarkup } from "../site/rca-workflow.ts";
import en from "../site/copy/en.json";
import zh from "../site/copy/zh.json";

describe("site language choice", () => {
  it("defaults to English unless ?lang=zh or saved choice is zh", () => {
    expect(pickLang("?lang=zh", "en", ["en-US"])).toBe("zh");
    expect(pickLang("?lang=en", "zh", ["zh-CN"])).toBe("en");
    expect(pickLang("?lang=fr", "zh", ["en-US"])).toBe("zh");
    expect(pickLang("", null, ["zh-TW", "en"])).toBe("en");
    expect(pickLang("", null, ["en-GB", "zh-CN"])).toBe("en");
    expect(pickLang("", "zh", ["en-US"])).toBe("zh");
    expect(pickLang("", "bogus", ["zh-CN"])).toBe("en");
  });

  it("looks up dotted keys and fills placeholders", () => {
    const copy = { a: { b: "{n} pellets" } };
    expect(lookup(copy, "a.b")).toBe("{n} pellets");
    expect(lookup(copy, "a.c")).toBe("a.c");
    expect(lookup(copy, "a")).toBe("a");
    expect(fill("{n} pellets, {m}", { n: 86 })).toBe("86 pellets, {m}");
  });

  it("both languages carry the same keys", () => {
    expect(flattenKeys(zh)).toEqual(flattenKeys(en));
  });

  it("every key referenced by CHAPTER_SPECS exists in both en and zh", () => {
    for (const ch of CHAPTER_SPECS) {
      for (const suffix of ["tab", "kicker", "title", "question", "takeaway"]) {
        const key = `${ch.i18nPrefix}.${suffix}`;
        expect(lookup(en, key)).not.toBe(key);
        expect(lookup(zh, key)).not.toBe(key);
      }
      for (const edge of ch.edges) {
        expect(lookup(en, edge.labelKey)).not.toBe(edge.labelKey);
        expect(lookup(zh, edge.labelKey)).not.toBe(edge.labelKey);
      }
      for (const node of ch.nodes) {
        expect(existsSync(node.sourcePath)).toBe(true);
        for (const suffix of ["title", "sub", "body", "io_in", "io_out"]) {
          const key = `${node.i18nPrefix}.${suffix}`;
          expect(lookup(en, key)).not.toBe(key);
          expect(lookup(zh, key)).not.toBe(key);
        }
      }
    }
  });
});

describe("ranking", () => {
  const others = [
    { id: "jev", pellets: 86 },
    { id: "ours", pellets: 176 },
    { id: "phi4_mini", pellets: 143 },
  ];
  it("places the player among the models, best first", () => {
    const r = rankPlayer(100, others);
    expect(r.order.map((e) => e.id)).toEqual(["ours", "phi4_mini", "you", "jev"]);
    expect(r.rank).toBe(3);
  });
  it("gives ties to the player and handles both ends", () => {
    expect(rankPlayer(176, others).rank).toBe(1);
    expect(rankPlayer(500, others).rank).toBe(1);
    expect(rankPlayer(0, others).rank).toBe(4);
  });
});

describe("model's-eyes mode", () => {
  function playing() {
    const s = createGame(100);
    while (s.phase !== "playing") step(s);
    return s;
  }

  it("holds at the first real junction until it is answered", () => {
    const s = playing();
    let held = false;
    for (let i = 0; i < 600 && !held; i++) {
      held = holdForPlayer(s, null);
      if (!held) step(s);
    }
    expect(held).toBe(true);
    const dp = findDecisionPoint(s);
    expect(dp.exits.length).toBeGreaterThan(1);
    expect(holdForPlayer(s, dp.key)).toBe(false);
  });

  it("shows the features encoding with one line per option", () => {
    const s = playing();
    const enc = encodeAhead(s);
    const text = compactState(enc.state);
    expect(JSON.parse(text)).toEqual(enc.state);
    for (const k of enc.keys) expect(text).toContain(`"${k}": {`);
  });
});

describe("mechanism & findings lab kernel", () => {
  const bank = buildScenarioBank(100);

  it("captures 4 distinct multi-exit junctions from seed 100", () => {
    const ids = ["opening-fork", "ghost-pinch", "power-pellet-chase", "tunnel-escape"] as const;
    const ticks = new Set<number>();
    const coords = new Set<string>();
    for (const id of ids) {
      const snap = bank[id];
      expect(snap.decision.exits.length).toBeGreaterThan(1);
      ticks.add(snap.tick);
      coords.add(`${snap.decision.x},${snap.decision.y}`);
    }
    expect(ticks.size).toBe(4);
    expect(coords.size).toBe(4);
  });

  it("inspects the /v1/systemone single-token contract and hardware latency", () => {
    const res = inspectSystemOneContract(bank["opening-fork"], "m3-ultra");
    expect(res.prompt).toContain("<|im_start|>system");
    expect(res.optionMap.length).toBeGreaterThan(1);
    expect(res.confidence).toBeGreaterThanOrEqual(0);
    expect(res.confidence).toBeLessThanOrEqual(1);
    expect(res.oneTokenTotalMs).toBeLessThan(res.chatSevenTokenMs);
  });

  it("uses matching 0.8B prefill and decode components for each measured server", () => {
    const m3 = inspectSystemOneContract(bank["opening-fork"], "m3-ultra");
    expect(m3.prefillMs).toBe(41);
    expect(m3.decodePerTokenMs).toBe(5.5);
    expect(m3.chatSevenTokenMs).toBe(74);

    const rtx = inspectSystemOneContract(bank["opening-fork"], "rtx-5090");
    expect(rtx.prefillMs).toBe(17);
    expect(rtx.decodePerTokenMs).toBe(2.1);
    expect(rtx.chatSevenTokenMs).toBe(30);
  });

  it("uses documentation-only IPv4 addresses in public RCA scenarios", () => {
    const addresses = JSON.stringify(RCA_CASES).match(/(?:\d{1,3}\.){3}\d{1,3}/g) ?? [];
    expect(addresses.length).toBeGreaterThan(0);
    for (const address of addresses) {
      expect(address).toMatch(/^(?:192\.0\.2|198\.51\.100|203\.0\.113)\.\d{1,3}$/);
    }
  });

  it("compares features, ascii-window, and ascii-full encoders and travel horizon", () => {
    const fast = inspectEncodersAndHorizon(bank["opening-fork"], 45);
    expect(fast.payloads.features.estimatedTokens).toBeLessThan(fast.payloads["ascii-full"].estimatedTokens);
    expect(fast.isStale).toBe(false);

    const slow = inspectEncodersAndHorizon(bank["opening-fork"], 290);
    expect(slow.tilesTraveledDuringCall).toBeGreaterThan(fast.tilesTraveledDuringCall);
    expect(slow.isStale).toBe(true);
  });

  it("computes teacher 5s rollout outcomes alongside student immediate features", () => {
    const gap = inspectDistillationGap(bank["tunnel-escape"], 5);
    expect(gap.options.length).toBeGreaterThan(1);
    for (const opt of gap.options) {
      expect(opt.studentFacts).toHaveProperty("pellets");
      expect(opt.teacherPeek).toHaveProperty("points");
    }
    expect(gap.informationGapExposed).toBe(true);
  });

  it("reproduces the vLLM raw vs processed_logprobs option-masking bug", () => {
    const vocab = { A: 0.49, C: 0.14, D: 0.12, B: 0.11 };
    const raw = inspectProcessedLogprobs(vocab, 3, 3, "raw");
    expect(raw.droppedValidLetters).toEqual(["B"]);

    const processed = inspectProcessedLogprobs(vocab, 3, 3, "processed");
    expect(processed.droppedValidLetters).toEqual([]);
    expect(Object.keys(processed.retainedOptions).sort()).toEqual(["A", "B", "C"]);
  });

  it("evaluates Section 18-24 ablation comparisons against 10-game standard error", () => {
    const rows = data.ablations as AblationRow[];
    const h100Vs5090 = compareAblationRun(rows, "pacman_08b_qwen_h100", "pacman_08b_qwen", "realtime");
    expect(h100Vs5090.withinNoise).toBe(true);

    const distilledVsBase = compareAblationRun(rows, "pacman_08b_qwen", "tev1_08b", "realtime");
    expect(distilledVsBase.withinNoise).toBe(false);
    expect(distilledVsBase.jevbenchDelta).toBeCloseTo(-0.13, 2);
  });

  it("inspects baremetal-rca synthetic and real OpenShift/Metal3 incidents", () => {
    const caseIds = Object.keys(RCA_CASES) as RcaCaseId[];
    expect(caseIds).toEqual(["synth-1000", "bmo-1098", "installer-10108", "ironic-137"]);

    const synth = inspectRcaCase("synth-1000");
    expect(synth.studentCorrect).toBe(true);
    expect(synth.deepseekCorrect).toBe(true);

    const bmo = inspectRcaCase("bmo-1098");
    expect(bmo.studentCorrect).toBe(true);
    expect(bmo.teacherAuditCorrect).toBe(false);

    const installer = inspectRcaCase("installer-10108");
    expect(installer.studentCorrect).toBe(false);
    expect(installer.deepseekCorrect).toBe(true);

    const ironic = inspectRcaCase("ironic-137");
    expect(ironic.studentCorrect).toBe(false);
    expect(ironic.deepseekCorrect).toBe(false);

    for (const id of caseIds) {
      const info = inspectRcaCase(id);
      expect(JSON.parse(info.formattedStateJson)).toEqual(info.snapshot.state);
      expect(lookup(en, info.snapshot.diagnosisKey)).not.toBe(info.snapshot.diagnosisKey);
      expect(lookup(zh, info.snapshot.diagnosisKey)).not.toBe(info.snapshot.diagnosisKey);
    }
  });
});


describe("dated evaluation report", () => {
  const translate = (copy: typeof en) => (key: string, vars?: Record<string, string | number>) => fill(lookup(copy, key), vars ?? {});

  it("keeps fresh results and historical baselines visible in both languages", () => {
    expect(evaluation.pacman.filter((row) => row.provenance === "fresh").map((row) => row.modelId)).toEqual(["gpt-6-luna", "jev-1.13.0", "jev-1.14.0", "responses:gpt-6-luna", "responses:gpt-6.1-terra"]);
    expect(evaluation.rca.filter((row) => row.provenance === "fresh").map((row) => row.modelId)).toEqual(["gpt-6-luna", "jev-1.13.0", "jev-1.14.0", "responses:gpt-6-luna", "responses:gpt-6.1-terra"]);
    expect(evaluation.pacman.filter((row) => row.provenance === "historical").map((row) => row.model)).toEqual(["GPT-6 Luna", "Jev", "pacman-0.8b-qwen"]);
    expect(evaluation.rca.filter((row) => row.provenance === "historical").map((row) => row.model)).toEqual(["rca-0.8b", "DeepSeek-V4-Flash"]);
    for (const copy of [en, zh]) {
      const { comparison: html, usage } = evaluationMarkup(evaluation, translate(copy));
      expect(html).toContain(copy.evaluation.fresh);
      expect(html).toContain(copy.evaluation.historical);
      expect(html).toContain(accuracyText(evaluation.majorityCorrect, evaluation.realN));
      expect(html).toContain("313/333");
      expect(html).toContain("11/48");
      expect(html).not.toContain("undefined");
      expect(html).not.toContain("evaluation.tasks.");
      expect(html).toContain(copy.evaluation.comparison_cost);
      expect(html).not.toContain(copy.evaluation.usage_title);
      expect(usage).toContain(copy.evaluation.usage_title);
      expect(html + usage).not.toContain("evaluation-total-cost");
    }
  });

  it("distinguishes incomplete accounting from a zero-cost complete run", () => {
    const row: UsageRow = { task: "pacman", modelId: "gpt-6-luna", model: "GPT-6 Luna", provider: "openai", requests: 4, inputTokens: 1000, cachedTokens: 0, cacheWriteTokens: 0, outputTokens: 0, reasoningTokens: null, costUsd: 0.0001, unknownUsageRequests: 1 };
    const pending: UsageRow = { ...row, task: "rca_real", requests: null, inputTokens: null, costUsd: null, unknownUsageRequests: null };
    expect(summarizeSpend([row, pending])).toEqual({ knownUsd: 0.0001, pendingTasks: 1, unknownRequests: 1, requests: 4 });
    const incomplete = evaluationMarkup({ ...evaluation, usage: [row] }, translate(en));
    expect(incomplete.comparison).toContain("lower bound");
    expect(incomplete.usage).toContain("lower bound");
    expect(incomplete.usage).not.toContain(en.evaluation.cost_complete);
    const complete = evaluationMarkup({ ...evaluation, usage: [{ ...row, costUsd: 0, unknownUsageRequests: 0 }] }, translate(en));
    expect(complete.comparison).toContain("$0.000000");
    expect(complete.usage).toContain("all with recorded usage");
    const missingRca = evaluationMarkup({ ...evaluation, pacman: [], usage: [row] }, translate(en));
    expect(missingRca.comparison).not.toContain("$0.000100");
    const pendingRca = evaluationMarkup({ ...evaluation, pacman: [], usage: [{ ...row, task: "rca_synthetic" }, pending] }, translate(en));
    expect(pendingRca.comparison).toContain(en.evaluation.pending);
    expect(pendingRca.comparison).not.toContain("$0.000100");
  });

  it("shows the task cost without preflight or historical same-model charges", () => {
    const row: UsageRow = { task: "pacman", modelId: "gpt-6-luna", model: "GPT-6 Luna", provider: "openai", requests: 1, inputTokens: 100, cachedTokens: 0, cacheWriteTokens: 0, outputTokens: 0, reasoningTokens: null, costUsd: 0.1, unknownUsageRequests: 0 };
    const { comparison } = evaluationMarkup({
      ...evaluation,
      usage: [row, { ...row, task: "rca_synthetic", costUsd: 0.2 }, { ...row, task: "rca_real", costUsd: 0.3 }, { ...row, task: "preflight", costUsd: 9 }],
    }, translate(en));
    const tables = comparison.split('</table>');
    expect(tables[0]).toContain("$0.100000");
    expect(tables[0]).not.toContain("$0.500000");
    expect(tables[1]).toContain("$0.500000");
    expect(tables[1]).not.toContain("$0.100000");
    expect(comparison).not.toContain("$9.");
    for (const row of comparison.matchAll(/<tr class="is-historical">(.+?)<\/tr>/g)) {
      expect(row[1]).toMatch(/<td>Not recorded<\/td>$/);
    }
  });

  it("totals both providers while keeping Jev cache telemetry not applicable", () => {
    const openai: UsageRow = { task: "pacman", modelId: "gpt-6-luna", model: "GPT-6 Luna", provider: "openai", requests: 10, inputTokens: 1000, cachedTokens: 0, cacheWriteTokens: 0, outputTokens: 0, reasoningTokens: null, costUsd: 0.0001, unknownUsageRequests: 0 };
    const jev: UsageRow = { ...openai, modelId: "jev-1.13.0", model: "jev-1.13.0", provider: "jev", requests: 20, inputTokens: 2000, cachedTokens: null, cacheWriteTokens: null, outputTokens: 40, costUsd: 0.000084 };
    expect(summarizeSpend([openai, jev])).toEqual({ knownUsd: 0.000184, pendingTasks: 0, unknownRequests: 0, requests: 30 });
    const { usage: html } = evaluationMarkup({ ...evaluation, usage: [openai, jev] }, translate(en));
    expect(html).not.toContain("evaluation-total-cost");
    expect(html).toContain("$0.000100");
    expect(html).toContain("$0.000084");
    expect(html).toContain('data-provider="openai"');
    expect(html).toContain('data-provider="jev"');
    expect(html).toMatch(/data-model="jev-1.13.0"[^]*?<td>N\/A<\/td><td>N\/A<\/td>/);
    expect(html).toContain("Jev USD = input × 0.042 / 1,000,000");
  });
});


describe("RCA workflow cost illustration", () => {
  const translate = (copy: typeof en) => (key: string, vars?: Record<string, string | number>) => fill(lookup(copy, key), vars ?? {});
  const defaults = { classificationShare: 10, replacementPercent: 10, extraPreparation: 5 };

  it("includes extra preparation and can exceed the baseline cost", () => {
    expect(scenarioCost(defaults)).toEqual({ unchanged: 90, classifier: 1, preparation: 5, total: 96, saving: 4 });
    const extra = { ...defaults, extraPreparation: 12 };
    expect(scenarioCost(extra).total).toBe(103);
    expect(scenarioCost(extra).saving).toBe(-3);
    expect(scenarioCost({ ...defaults, replacementPercent: 0, extraPreparation: 0 }).saving).toBe(10);
    for (const copy of [en, zh]) {
      const html = workflowCostMarkup(extra, translate(copy));
      expect(html).toContain(fill(copy.workflow.extra, { n: 3 }));
      expect(html).toContain(fill(copy.workflow.units, { n: 103 }));
      expect(html).not.toContain("undefined");
    }
  });

  it("keeps workflow assumptions separate from measured results in both languages", () => {
    for (const copy of [en, zh]) {
      const html = workflowMarkup(translate(copy), defaults);
      expect(html).toContain(copy.workflow.illustration);
      expect(html).toContain(copy.workflow.measured_scope);
      expect(html).toContain(copy.workflow.context_limits);
      for (const stage of Object.values(copy.workflow.stages)) expect(html).toContain(stage.title);
      for (const key of Object.keys(defaults)) expect(html).toContain(`id="workflow-${key}"`);
      expect(html).not.toContain("workflow.controls.");
    }
  });
});
