import {
  CHAPTER_SPECS,
  compareAblationRun,
  inspectDistillationGap,
  inspectEncodersAndHorizon,
  inspectProcessedLogprobs,
  inspectRcaCase,
  inspectSystemOneContract,
  type AblationRow,
  type AblationVariantId,
  type ChapterId,
  type ChapterSpec,
  type EncoderId,
  type HardwareId,
  type RcaCaseId,
  type ScenarioBank,
  type ScenarioId,
} from "./lab.ts";

export interface MechanismLabOptions {
  readonly scenarioBank: ScenarioBank;
  readonly ablations: readonly AblationRow[];
  readonly repoUrl: string;
  readonly t: (key: string, vars?: Record<string, string | number>) => string;
}

export interface MechanismLabController {
  refresh(): void;
  selectChapter(id: ChapterId): void;
  selectNode(id: string): void;
  selectScenario(id: ScenarioId): void;
  selectRcaCase(id: RcaCaseId): void;
  reset(): void;
  getState(): {
    activeChapterId: ChapterId;
    selectedNodeId: string;
    scenarioId: ScenarioId;
    rcaCaseId: RcaCaseId;
    hardware: HardwareId;
    encoderId: EncoderId;
    latencyMs: number;
    horizonSeconds: number;
    logprobMode: "raw" | "processed";
    ablationCandidateId: AblationVariantId;
    ablationClock: "realtime" | "lockstep";
  };
}

const SCENARIO_IDS: readonly ScenarioId[] = [
  "opening-fork",
  "ghost-pinch",
  "power-pellet-chase",
  "tunnel-escape",
];

const RCA_CASE_IDS: readonly RcaCaseId[] = [
  "synth-1000",
  "bmo-1098",
  "installer-10108",
  "ironic-137",
];

const HARDWARE_IDS: readonly HardwareId[] = ["m3-ultra", "rtx-5090", "h100-vllm"];
const ENCODER_IDS: readonly EncoderId[] = ["features", "ascii-window", "ascii-full"];
const HORIZON_OPTIONS: readonly number[] = [1, 3, 5, 8];

const SAMPLE_VOCAB_PROBS: Record<string, number> = {
  A: 0.49,
  C: 0.14,
  D: 0.12,
  B: 0.11,
  "\\n": 0.08,
  E: 0.06,
};

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
}

function formatRichText(s: string): string {
  return escapeHtml(s)
    .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
    .replace(/&lt;b&gt;([\s\S]*?)&lt;\/b&gt;/g, "<b>$1</b>");
}

function isChapterId(v: string): v is ChapterId {
  return CHAPTER_SPECS.some((c) => c.id === v);
}

export function mountMechanismLab(host: HTMLElement, opts: MechanismLabOptions): MechanismLabController {
  const { scenarioBank, ablations, repoUrl, t } = opts;

  const initialHash = location.hash.replace(/^#/, "");
  let activeChapterId: ChapterId = isChapterId(initialHash) ? initialHash : "ch5-rca";
  let selectedNodeId: string = CHAPTER_SPECS.find((c) => c.id === activeChapterId)!.nodes[0].id;
  let scenarioId: ScenarioId = "tunnel-escape";
  let rcaCaseId: RcaCaseId = "installer-10108";
  let hardware: HardwareId = "m3-ultra";
  let encoderId: EncoderId = "features";
  let latencyMs = 55;
  let horizonSeconds = 5;
  let logprobMode: "raw" | "processed" = "raw";
  let ablationCandidateId: AblationVariantId = "pacman_08b_mix_h100";
  let ablationClock: "realtime" | "lockstep" = "realtime";

  function currentChapter(): ChapterSpec {
    return CHAPTER_SPECS.find((c) => c.id === activeChapterId) ?? CHAPTER_SPECS[0];
  }

  function renderDiagramSvg(chapter: ChapterSpec): string {
    const nodeMap = new Map(chapter.nodes.map((n) => [n.id, n]));

    const edgesSvg = chapter.edges
      .map((edge) => {
        const from = nodeMap.get(edge.from);
        const to = nodeMap.get(edge.to);
        if (!from || !to) return "";
        const x1 = from.x + from.width;
        const y1 = from.y + Math.round(from.height / 2);
        const x2 = to.x - 6;
        const y2 = to.y + Math.round(to.height / 2);
        const midX = Math.round((x1 + x2) / 2);
        const rawLabel = t(edge.labelKey);
        const label = escapeHtml(rawLabel);
        const pillW = Math.max(42, rawLabel.length * 6.5 + 14);
        return `
          <g class="lab-edge">
            <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#146c3b" stroke-width="2" marker-end="url(#lab-arrow)" />
            <rect x="${Math.round(midX - pillW / 2)}" y="${y1 - 26}" width="${Math.round(pillW)}" height="18" rx="9" fill="#eef7f0" stroke="#b9dfc2" stroke-width="1" />
            <text x="${midX}" y="${y1 - 14}" text-anchor="middle" class="lab-edge-label">${label}</text>
          </g>
        `;
      })
      .join("");

    const nodesSvg = chapter.nodes
      .map((node) => {
        const isSelected = node.id === selectedNodeId;
        const title = escapeHtml(t(`${node.i18nPrefix}.title`));
        const sub = escapeHtml(t(`${node.i18nPrefix}.sub`));
        const roleLabel = escapeHtml(t(`lab.roles.${node.role}`));
        return `
          <g class="lab-node ${isSelected ? "is-selected" : ""} role-${node.role}" data-lab-node="${node.id}" tabindex="0" role="button" aria-pressed="${isSelected}">
            <rect class="lab-node-card" x="${node.x}" y="${node.y}" width="${node.width}" height="${node.height}" rx="14" />
            <rect class="lab-node-bar" x="${node.x + 12}" y="${node.y + 10}" width="34" height="20" rx="6" />
            <text x="${node.x + 29}" y="${node.y + 24}" text-anchor="middle" class="lab-node-step">${escapeHtml(node.step)}</text>
            <text x="${node.x + node.width - 12}" y="${node.y + 24}" text-anchor="end" class="lab-node-role">${roleLabel}</text>
            <text x="${node.x + 14}" y="${node.y + 54}" class="lab-node-title">${title}</text>
            <text x="${node.x + 14}" y="${node.y + 76}" class="lab-node-sub">${sub}</text>
            <text x="${node.x + 14}" y="${node.y + 98}" class="lab-node-path">${escapeHtml(node.sourcePath)}</text>
          </g>
        `;
      })
      .join("");

    return `
      <svg class="lab-svg" viewBox="0 0 1000 205" role="img" aria-label="${escapeHtml(t(`${chapter.i18nPrefix}.title`))}">
        <defs>
          <pattern id="lab-grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <circle cx="2" cy="2" r="1" fill="#dcd5ce" />
          </pattern>
          <marker id="lab-arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 8 5 L 0 9 z" fill="#146c3b" />
          </marker>
        </defs>
        <rect x="2" y="2" width="996" height="201" rx="16" fill="url(#lab-grid)" stroke="#e8e2dc" stroke-width="1" />
        <text x="20" y="28" class="lab-svg-caption">${escapeHtml(t("lab.ui.diagram_hint"))}</text>
        ${nodesSvg}
        ${edgesSvg}
      </svg>
    `;
  }

  function renderInspector(chapter: ChapterSpec): string {
    const node = chapter.nodes.find((n) => n.id === selectedNodeId) ?? chapter.nodes[0];
    const sourceHref = `${repoUrl}/blob/master/${node.sourcePath}`;
    return `
      <div class="lab-inspector-head">
        <div class="lab-inspector-badges">
          <span class="lab-badge lab-badge-step">${escapeHtml(node.step)}</span>
          <span class="lab-badge lab-badge-role role-${node.role}">${escapeHtml(t(`lab.roles.${node.role}`))}</span>
          <h4 class="lab-inspector-title">${formatRichText(t(`${node.i18nPrefix}.title`))}</h4>
        </div>
        <a class="lab-source-link" href="${escapeHtml(sourceHref)}" target="_blank" rel="noopener">
          <code>${escapeHtml(node.sourcePath)}</code> ↗
        </a>
      </div>
      <p class="lab-inspector-body">${formatRichText(t(`${node.i18nPrefix}.body`))}</p>
      <div class="lab-io-row">
        <span class="lab-io-pill"><b>${escapeHtml(t("lab.ui.input_label"))}:</b> ${formatRichText(t(`${node.i18nPrefix}.io_in`))}</span>
        <span class="lab-io-arrow" aria-hidden="true">→</span>
        <span class="lab-io-pill"><b>${escapeHtml(t("lab.ui.output_label"))}:</b> ${formatRichText(t(`${node.i18nPrefix}.io_out`))}</span>
      </div>
      <pre class="lab-code"><code>${escapeHtml(node.codeSnippet)}</code></pre>
    `;
  }

  function renderScenarioBar(): string {
    const snap = scenarioBank[scenarioId];
    return `
      <div class="lab-scenario-bar">
        <span class="lab-control-label">${escapeHtml(t("lab.ui.scenario_label"))}</span>
        <div class="lab-pill-group" role="group" aria-label="${escapeHtml(t("lab.ui.scenario_label"))}">
          ${SCENARIO_IDS.map((id) => {
            const s = scenarioBank[id];
            const active = id === scenarioId;
            return `<button type="button" class="lab-pill ${active ? "is-active" : ""}" data-lab-scenario="${id}">
              ${escapeHtml(t(`lab.scenarios.${id}`))}
              <small>(${s.decision.x},${s.decision.y})</small>
            </button>`;
          }).join("")}
        </div>
        <span class="lab-scenario-meta">
          ${escapeHtml(t("lab.ui.junction_meta", { x: snap.decision.x, y: snap.decision.y, dist: Math.round(snap.decision.distance * 10) / 10, exits: snap.decision.exits.length }))}
        </span>
      </div>
    `;
  }

  function renderChapter1Workbench(): string {
    const snap = scenarioBank[scenarioId];
    const info = inspectSystemOneContract(snap, hardware);
    const maxMs = Math.max(info.chatSevenTokenMs, 120);
    const sysPct = Math.min(100, Math.round((info.oneTokenTotalMs / maxMs) * 100));
    const chatPct = Math.min(100, Math.round((info.chatSevenTokenMs / maxMs) * 100));

    return `
      ${renderScenarioBar()}
      <div class="lab-wb-grid">
        <div class="lab-wb-panel">
          <div class="lab-panel-head">
            <h5>${escapeHtml(t("lab.ch1.wb_latency_title"))}</h5>
            <div class="lab-pill-group">
              ${HARDWARE_IDS.map(
                (hw) =>
                  `<button type="button" class="lab-pill lab-pill-sm ${hw === hardware ? "is-active" : ""}" data-lab-hw="${hw}">${escapeHtml(t(`lab.hardware.${hw}`))}</button>`,
              ).join("")}
            </div>
          </div>
          <div class="lab-bars">
            <div class="lab-bar-row">
              <div class="lab-bar-meta">
                <span>${escapeHtml(t("lab.ch1.one_token_label"))}</span>
                <b class="lab-mono">${info.oneTokenTotalMs} ms</b>
              </div>
              <div class="lab-bar-track"><div class="lab-bar-fill is-strong" style="width:${sysPct}%"></div></div>
            </div>
            <div class="lab-bar-row">
              <div class="lab-bar-meta">
                <span>${escapeHtml(t("lab.ch1.seven_token_label"))}</span>
                <b class="lab-mono">${info.chatSevenTokenMs} ms</b>
              </div>
              <div class="lab-bar-track"><div class="lab-bar-fill is-muted" style="width:${chatPct}%"></div></div>
            </div>
          </div>
          <p class="lab-panel-note">${escapeHtml(t("lab.ch1.latency_note", { prefill: info.prefillMs, decode: info.decodePerTokenMs }))}</p>
        </div>

        <div class="lab-wb-panel">
          <div class="lab-panel-head">
            <h5>${escapeHtml(t("lab.ch1.wb_probs_title"))}</h5>
            <span class="lab-badge lab-badge-strong">${escapeHtml(t("lab.ch1.confidence_badge", { conf: info.confidence.toFixed(2) }))}</span>
          </div>
          <div class="lab-bars">
            ${info.optionMap
              .map((opt) => {
                const pct = Math.round(opt.probability * 100);
                const isWinner = opt.key === info.chosenKey;
                return `
                  <div class="lab-bar-row">
                    <div class="lab-bar-meta">
                      <span><code class="lab-opt-code">${escapeHtml(opt.letter)}</code> → <b>${escapeHtml(opt.key)}</b></span>
                      <b class="lab-mono">${pct}%</b>
                    </div>
                    <div class="lab-bar-track">
                      <div class="lab-bar-fill ${isWinner ? "is-strong" : "is-plain"}" style="width:${Math.max(2, pct)}%"></div>
                    </div>
                  </div>
                `;
              })
              .join("")}
          </div>
          <details class="lab-prompt-details">
            <summary>${escapeHtml(t("lab.ch1.prompt_toggle"))}</summary>
            <pre class="lab-code lab-code-compact"><code>${escapeHtml(info.prompt)}</code></pre>
          </details>
        </div>
      </div>
    `;
  }

  function renderChapter2Workbench(): string {
    const snap = scenarioBank[scenarioId];
    const info = inspectEncodersAndHorizon(snap, latencyMs);
    const activePayload = info.payloads[encoderId];

    return `
      ${renderScenarioBar()}
      <div class="lab-wb-grid">
        <div class="lab-wb-panel">
          <div class="lab-panel-head">
            <h5>${escapeHtml(t("lab.ch2.wb_encoder_title"))}</h5>
            <div class="lab-pill-group">
              ${ENCODER_IDS.map(
                (id) =>
                  `<button type="button" class="lab-pill lab-pill-sm ${id === encoderId ? "is-active" : ""}" data-lab-encoder="${id}">${escapeHtml(id)}</button>`,
              ).join("")}
            </div>
          </div>
          <div class="lab-stat-strip">
            <div class="lab-stat-cell">
              <span>${escapeHtml(t("lab.ch2.tokens_label"))}</span>
              <b>~${activePayload.estimatedTokens}</b>
            </div>
            <div class="lab-stat-cell">
              <span>${escapeHtml(t("lab.ch2.chars_label"))}</span>
              <b>${activePayload.chars}</b>
            </div>
            <div class="lab-stat-cell">
              <span>${escapeHtml(t("lab.ch2.conf_band_label"))}</span>
              <b>${escapeHtml(activePayload.observedConfidenceBand)}</b>
            </div>
          </div>
          <pre class="lab-code lab-code-scroll"><code>${escapeHtml(activePayload.text)}</code></pre>
        </div>

        <div class="lab-wb-panel">
          <div class="lab-panel-head">
            <h5>${escapeHtml(t("lab.ch2.wb_horizon_title"))}</h5>
            <span data-lab-horizon-status class="lab-badge ${info.isStale ? "lab-badge-warn" : "lab-badge-strong"}">
              ${escapeHtml(t(info.isStale ? "lab.ch2.status_stale" : "lab.ch2.status_fresh"))}
            </span>
          </div>
          <label class="lab-slider-wrap" for="lab-latency-slider">
            <div class="lab-slider-head">
              <span>${escapeHtml(t("lab.ch2.slider_label"))}</span>
              <b data-lab-latency-value class="lab-mono">${info.latencyMs} ms</b>
            </div>
            <input id="lab-latency-slider" type="range" min="20" max="300" step="5" value="${info.latencyMs}" />
          </label>
          <div class="lab-stat-strip">
            <div class="lab-stat-cell">
              <span>${escapeHtml(t("lab.ch2.speed_label"))}</span>
              <b>${info.pacSpeedTilesPerSec} tiles/s</b>
            </div>
            <div class="lab-stat-cell">
              <span>${escapeHtml(t("lab.ch2.traveled_label"))}</span>
              <b data-lab-horizon-traveled>${info.tilesTraveledDuringCall} tiles</b>
            </div>
            <div class="lab-stat-cell">
              <span>${escapeHtml(t("lab.ch2.junction_dist_label"))}</span>
              <b>${info.junctionDistanceTiles} tiles</b>
            </div>
          </div>
          <p data-lab-horizon-note class="lab-panel-note">
            ${escapeHtml(
              t(info.isStale ? "lab.ch2.horizon_stale_body" : "lab.ch2.horizon_fresh_body", {
                traveled: info.tilesTraveledDuringCall,
                dist: info.junctionDistanceTiles,
                margin: Math.abs(info.arrivalMarginTiles),
              }),
            )}
          </p>
        </div>
      </div>
    `;
  }

  function updateHorizon(): void {
    const info = inspectEncodersAndHorizon(scenarioBank[scenarioId], latencyMs);
    const status = host.querySelector<HTMLElement>("[data-lab-horizon-status]");
    if (status) {
      status.textContent = t(info.isStale ? "lab.ch2.status_stale" : "lab.ch2.status_fresh");
      status.classList.toggle("lab-badge-warn", info.isStale);
      status.classList.toggle("lab-badge-strong", !info.isStale);
    }
    const outputs = {
      "data-lab-latency-value": `${info.latencyMs} ms`,
      "data-lab-horizon-traveled": `${info.tilesTraveledDuringCall} tiles`,
      "data-lab-horizon-note": t(info.isStale ? "lab.ch2.horizon_stale_body" : "lab.ch2.horizon_fresh_body", {
        traveled: info.tilesTraveledDuringCall,
        dist: info.junctionDistanceTiles,
        margin: Math.abs(info.arrivalMarginTiles),
      }),
    };
    for (const [attribute, text] of Object.entries(outputs)) {
      const output = host.querySelector<HTMLElement>(`[${attribute}]`);
      if (output) output.textContent = text;
    }
  }

  function renderChapter3Workbench(): string {
    const snap = scenarioBank[scenarioId];
    const info = inspectDistillationGap(snap, horizonSeconds);

    return `
      ${renderScenarioBar()}
      <div class="lab-wb-panel">
        <div class="lab-panel-head">
          <div>
            <h5>${escapeHtml(t("lab.ch3.wb_title"))}</h5>
            <p class="lab-panel-sub">
              ${escapeHtml(
                t("lab.ch3.moves_summary", {
                  student: info.studentGreedyMove,
                  teacher: info.teacherLookaheadMove,
                }),
              )}
            </p>
          </div>
          <div class="lab-pill-group">
            ${HORIZON_OPTIONS.map(
              (sec) =>
                `<button type="button" class="lab-pill lab-pill-sm ${sec === horizonSeconds ? "is-active" : ""}" data-lab-horizon="${sec}">${sec}s ${escapeHtml(t("lab.ch3.horizon_unit"))}</button>`,
            ).join("")}
          </div>
        </div>
        <div class="lab-table-wrap">
          <table class="lab-table">
            <thead>
              <tr>
                <th>${escapeHtml(t("lab.ch3.col_option"))}</th>
                <th>${escapeHtml(t("lab.ch3.col_student"))}</th>
                <th>${escapeHtml(t("lab.ch3.col_teacher", { sec: horizonSeconds }))}</th>
                <th>${escapeHtml(t("lab.ch3.col_value"))}</th>
              </tr>
            </thead>
            <tbody>
              ${info.options
                .map((opt) => {
                  const isTeacherPick = opt.key === info.teacherLookaheadMove;
                  const died = opt.teacherPeek.dies_after_s !== null;
                  return `
                    <tr class="${isTeacherPick ? "is-winner-row" : ""} ${died ? "is-danger-row" : ""}">
                      <td>
                        <code class="lab-opt-code">${escapeHtml(opt.letter)}</code>
                        <b>${escapeHtml(opt.key)}</b>
                        ${isTeacherPick ? `<span class="lab-badge lab-badge-strong">${escapeHtml(t("lab.ch3.teacher_pick"))}</span>` : ""}
                      </td>
                      <td><code class="lab-inline-json">${escapeHtml(JSON.stringify(opt.studentFacts))}</code></td>
                      <td>
                        <code class="lab-inline-json">${escapeHtml(JSON.stringify(opt.teacherPeek))}</code>
                      </td>
                      <td class="lab-mono"><b>${opt.rolloutScore}</b></td>
                    </tr>
                  `;
                })
                .join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  function renderChapter4Workbench(): string {
    const maskInfo = inspectProcessedLogprobs(SAMPLE_VOCAB_PROBS, 3, 3, logprobMode);
    const cmp = compareAblationRun(ablations, ablationCandidateId, "pacman_08b_qwen", ablationClock);
    const sign = cmp.deltaPellets >= 0 ? "+" : "";

    return `
      <div class="lab-wb-grid">
        <div class="lab-wb-panel">
          <div class="lab-panel-head">
            <h5>${formatRichText(t("lab.ch4.wb_mask_title"))}</h5>
            <div class="lab-pill-group">
              <button type="button" class="lab-pill lab-pill-sm ${logprobMode === "raw" ? "is-active" : ""}" data-lab-logprob-mode="raw">
                ${escapeHtml(t("lab.ch4.mode_raw"))}
              </button>
              <button type="button" class="lab-pill lab-pill-sm ${logprobMode === "processed" ? "is-active" : ""}" data-lab-logprob-mode="processed">
                ${escapeHtml(t("lab.ch4.mode_processed"))}
              </button>
            </div>
          </div>
          <p class="lab-panel-note">${escapeHtml(t("lab.ch4.mask_explain"))}</p>
          <div class="lab-bars">
            ${maskInfo.rawTopK
              .map((entry) => {
                const pct = Math.round(entry.prob * 100);
                return `
                  <div class="lab-bar-row">
                    <div class="lab-bar-meta">
                      <span>
                        <code class="lab-opt-code">${escapeHtml(entry.token)}</code>
                        ${entry.isOption ? `<span class="lab-badge lab-badge-strong">${escapeHtml(t("lab.ch4.valid_opt"))}</span>` : `<span class="lab-badge lab-badge-warn">${escapeHtml(t("lab.ch4.invalid_opt"))}</span>`}
                      </span>
                      <b class="lab-mono">${pct}%</b>
                    </div>
                    <div class="lab-bar-track">
                      <div class="lab-bar-fill ${entry.isOption ? "is-strong" : "is-warn"}" style="width:${Math.max(4, pct)}%"></div>
                    </div>
                  </div>
                `;
              })
              .join("")}
          </div>
          <div class="lab-callout-mini ${maskInfo.droppedValidLetters.length > 0 ? "is-warn" : "is-ok"}">
            ${
              maskInfo.droppedValidLetters.length > 0
                ? escapeHtml(t("lab.ch4.dropped_warning", { dropped: maskInfo.droppedValidLetters.join(", ") }))
                : escapeHtml(t("lab.ch4.retained_ok", { retained: Object.keys(maskInfo.retainedOptions).join(", ") }))
            }
          </div>
        </div>

        <div class="lab-wb-panel">
          <div class="lab-panel-head">
            <h5>${escapeHtml(t("lab.ch4.wb_ablation_title"))}</h5>
            <div class="lab-pill-group">
              <button type="button" class="lab-pill lab-pill-sm ${ablationClock === "realtime" ? "is-active" : ""}" data-lab-clock="realtime">
                ${escapeHtml(t("lab.ch4.clock_realtime"))}
              </button>
              <button type="button" class="lab-pill lab-pill-sm ${ablationClock === "lockstep" ? "is-active" : ""}" data-lab-clock="lockstep">
                ${escapeHtml(t("lab.ch4.clock_lockstep"))}
              </button>
            </div>
          </div>
          <div class="lab-table-wrap">
            <table class="lab-table lab-table-compact">
              <thead>
                <tr>
                  <th>${escapeHtml(t("lab.ch4.col_variant"))}</th>
                  <th>${escapeHtml(t("lab.ch4.col_teacher"))}</th>
                  <th>${escapeHtml(t("lab.ch4.col_pellets"))}</th>
                  <th>JevBench</th>
                </tr>
              </thead>
              <tbody>
                ${ablations
                  .map((row) => {
                    const isSelected = row.id === ablationCandidateId;
                    return `
                      <tr class="lab-clickable-row ${isSelected ? "is-winner-row" : ""}" data-lab-variant="${row.id}">
                        <td><code>${escapeHtml(row.id)}</code></td>
                        <td>${escapeHtml(row.teacher)}</td>
                        <td class="lab-mono"><b>${row[ablationClock]}</b> <small>±${row.se}</small></td>
                        <td class="lab-mono">${row.jevbench.toFixed(2)}</td>
                      </tr>
                    `;
                  })
                  .join("")}
              </tbody>
            </table>
          </div>
          <div class="lab-callout-mini ${cmp.withinNoise ? "is-ok" : "is-warn"}">
            <b>${escapeHtml(cmp.candidate.id)}</b> vs <b>${escapeHtml(cmp.baseline.id)}</b>:
            <span class="lab-mono">${sign}${cmp.deltaPellets}</span> ${escapeHtml(t("lab.ch4.pellets_unit"))}
            (SE ±${cmp.combinedSe}, z = ${cmp.zScore}).
            <b>${escapeHtml(t(cmp.withinNoise ? "lab.ch4.verdict_noise" : "lab.ch4.verdict_sig"))}</b>
          </div>
        </div>
      </div>
    `;
  }

  function renderChapter5Workbench(): string {
    const info = inspectRcaCase(rcaCaseId);
    const snap = info.snapshot;

    return `
      <div class="lab-scenario-bar">
        <span class="lab-control-label">${escapeHtml(t("lab.ch5.case_label"))}</span>
        <div class="lab-pill-group" role="group" aria-label="${escapeHtml(t("lab.ch5.case_label"))}">
          ${RCA_CASE_IDS.map((id) => {
            const active = id === rcaCaseId;
            return `<button type="button" class="lab-pill ${active ? "is-active" : ""}" data-lab-rca-case="${id}">
              ${escapeHtml(t(`lab.ch5.cases.${id}`))}
            </button>`;
          }).join("")}
        </div>
        <span class="lab-scenario-meta">
          <code>${escapeHtml(snap.source)}</code> · <b>${escapeHtml(t(`lab.ch5.splits.${snap.split}`))}</b>
        </span>
      </div>

      <div class="lab-wb-grid">
        <div class="lab-wb-panel">
          <div class="lab-panel-head">
            <h5>${formatRichText(t("lab.ch5.wb_payload_title"))}</h5>
            <span class="lab-badge ${info.studentCorrect ? "lab-badge-strong" : "lab-badge-warn"}">
              ${escapeHtml(t("lab.ch5.truth_badge", { label: snap.humanLabel }))}
            </span>
          </div>
          <div class="lab-stat-strip">
            <div class="lab-stat-cell">
              <span>${escapeHtml(t("lab.ch5.student_pick_label"))}</span>
              <b><code>${escapeHtml(snap.studentChoice)}</code> ${info.studentCorrect ? "✓" : "✗"}</b>
            </div>
            <div class="lab-stat-cell">
              <span>${escapeHtml(t("lab.ch5.deepseek_pick_label"))}</span>
              <b><code>${escapeHtml(snap.deepseekZeroShot)}</code> ${info.deepseekCorrect ? "✓" : "✗"}</b>
            </div>
            <div class="lab-stat-cell">
              <span>${escapeHtml(t("lab.ch5.teacher_audit_label"))}</span>
              <b><code>${escapeHtml(snap.teacherWithComments)}</code> ${info.teacherAuditCorrect ? "✓" : "✗"}</b>
            </div>
          </div>
          <pre class="lab-code lab-code-scroll"><code>${escapeHtml(info.formattedStateJson)}</code></pre>
        </div>

        <div class="lab-wb-panel">
          <div class="lab-panel-head">
            <h5>${formatRichText(t("lab.ch5.wb_probs_title", { ms: snap.studentLatencyMs }))}</h5>
            <span class="lab-badge ${info.studentCorrect ? "lab-badge-strong" : "lab-badge-warn"}">
              ${escapeHtml(t(info.studentCorrect ? "lab.ch5.status_match" : "lab.ch5.status_gap"))}
            </span>
          </div>
          <div class="lab-bars">
            ${snap.topProbs
              .map((entry) => {
                const pct = Math.max(1, Math.round(entry.prob * 100));
                const isTruth = entry.option === snap.humanLabel;
                return `
                  <div class="lab-bar-row">
                    <div class="lab-bar-meta">
                      <span>
                        <code>${escapeHtml(entry.option)}</code>
                        ${isTruth ? `<span class="lab-badge lab-badge-strong">${escapeHtml(t("lab.ch5.ground_truth_tag"))}</span>` : ""}
                      </span>
                      <b class="lab-mono">${(entry.prob * 100).toFixed(1)}%</b>
                    </div>
                    <div class="lab-bar-track">
                      <div class="lab-bar-fill ${isTruth ? "is-strong" : "is-warn"}" style="width:${Math.max(4, pct)}%"></div>
                    </div>
                  </div>
                `;
              })
              .join("")}
          </div>
          <div class="lab-callout-mini ${info.studentCorrect ? "is-ok" : "is-warn"}">
            ${formatRichText(t(snap.diagnosisKey))}
          </div>
          <div class="lab-stat-strip">
            <div class="lab-stat-cell">
              <span>${escapeHtml(t("lab.ch5.prior_synth_label"))}</span>
              <b>4.3% <code>product_bug</code></b>
            </div>
            <div class="lab-stat-cell">
              <span>${escapeHtml(t("lab.ch5.prior_real_label"))}</span>
              <b>70.8% <code>product_bug</code></b>
            </div>
            <div class="lab-stat-cell">
              <span>${escapeHtml(t("lab.ch5.config_agree_label"))}</span>
              <b>5 / 14 (35.7%)</b>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  function renderWorkbench(chapter: ChapterSpec): string {
    switch (chapter.id) {
      case "ch1-contract":
        return renderChapter1Workbench();
      case "ch2-encoding":
        return renderChapter2Workbench();
      case "ch3-distill":
        return renderChapter3Workbench();
      case "ch4-ablations":
        return renderChapter4Workbench();
      case "ch5-rca":
        return renderChapter5Workbench();
    }
  }

  function render(): void {
    const chapter = currentChapter();
    host.innerHTML = `
      <div class="lab-shell">
        <div class="lab-nav" role="tablist" aria-label="${escapeHtml(t("lab.title"))}">
          ${CHAPTER_SPECS.map((c) => {
            const active = c.id === activeChapterId;
            const num = String(c.index).padStart(2, "0");
            return `
              <button
                type="button"
                role="tab"
                aria-selected="${active}"
                class="lab-tab ${active ? "is-active" : ""}"
                data-lab-chapter="${c.id}"
              >
                <span class="lab-tab-num">${num}</span>
                <span class="lab-tab-label">${escapeHtml(t(`${c.i18nPrefix}.tab`))}</span>
              </button>
            `;
          }).join("")}
        </div>

        <div class="card lab-card">
          <div class="lab-chapter-head">
            <div>
              <span class="lab-chapter-kicker">${escapeHtml(t(`${chapter.i18nPrefix}.kicker`))}</span>
              <h3 class="subheading lab-chapter-title">${formatRichText(t(`${chapter.i18nPrefix}.title`))}</h3>
              <p class="lab-chapter-question">${formatRichText(t(`${chapter.i18nPrefix}.question`))}</p>
            </div>
            <button type="button" class="btn btn-ghost btn-small" id="lab-reset">
              ${escapeHtml(t("lab.ui.reset"))}
            </button>
          </div>

          <div class="lab-diagram-wrap" id="lab-diagram">
            ${renderDiagramSvg(chapter)}
          </div>

          <div class="lab-inspector" id="lab-inspector">
            ${renderInspector(chapter)}
          </div>

          <div class="lab-workbench" id="lab-workbench">
            ${renderWorkbench(chapter)}
          </div>

          <div class="lab-takeaway" id="lab-takeaway">
            <span class="lab-takeaway-label">${escapeHtml(t("lab.ui.takeaway_label"))}</span>
            <p class="lab-takeaway-body">${formatRichText(t(`${chapter.i18nPrefix}.takeaway`))}</p>
          </div>
        </div>
      </div>
    `;
  }

  function selectChapter(id: ChapterId): void {
    if (!isChapterId(id)) return;
    activeChapterId = id;
    const spec = currentChapter();
    selectedNodeId = spec.nodes[0].id;
    try {
      const url = new URL(location.href);
      url.hash = id;
      history.replaceState(null, "", url);
    } catch {
      // Ignore history errors in restricted environments.
    }
    render();
  }

  function selectNode(id: string): void {
    const spec = currentChapter();
    if (spec.nodes.some((n) => n.id === id)) {
      selectedNodeId = id;
      render();
    }
  }

  function selectScenario(id: ScenarioId): void {
    if (SCENARIO_IDS.includes(id)) {
      scenarioId = id;
      render();
    }
  }

  function selectRcaCase(id: RcaCaseId): void {
    if (RCA_CASE_IDS.includes(id)) {
      rcaCaseId = id;
      render();
    }
  }

  function reset(): void {
    activeChapterId = "ch5-rca";
    selectedNodeId = CHAPTER_SPECS.find((c) => c.id === "ch5-rca")!.nodes[0].id;
    scenarioId = "tunnel-escape";
    rcaCaseId = "installer-10108";
    hardware = "m3-ultra";
    encoderId = "features";
    latencyMs = 55;
    horizonSeconds = 5;
    logprobMode = "raw";
    ablationCandidateId = "pacman_08b_mix_h100";
    ablationClock = "realtime";
    render();
  }

  host.addEventListener("click", (ev) => {
    const target = ev.target as Element | null;
    if (!target) return;

    const chBtn = target.closest<HTMLElement>("[data-lab-chapter]");
    if (chBtn && chBtn.dataset.labChapter) {
      selectChapter(chBtn.dataset.labChapter as ChapterId);
      return;
    }

    const nodeEl = target.closest<HTMLElement | SVGElement>("[data-lab-node]");
    if (nodeEl) {
      const nodeId = nodeEl.getAttribute("data-lab-node");
      if (nodeId) selectNode(nodeId);
      return;
    }

    const scBtn = target.closest<HTMLElement>("[data-lab-scenario]");
    if (scBtn && scBtn.dataset.labScenario) {
      selectScenario(scBtn.dataset.labScenario as ScenarioId);
      return;
    }

    const rcaBtn = target.closest<HTMLElement>("[data-lab-rca-case]");
    if (rcaBtn && rcaBtn.dataset.labRcaCase) {
      selectRcaCase(rcaBtn.dataset.labRcaCase as RcaCaseId);
      return;
    }

    const hwBtn = target.closest<HTMLElement>("[data-lab-hw]");
    if (hwBtn && hwBtn.dataset.labHw) {
      hardware = hwBtn.dataset.labHw as HardwareId;
      render();
      return;
    }

    const encBtn = target.closest<HTMLElement>("[data-lab-encoder]");
    if (encBtn && encBtn.dataset.labEncoder) {
      encoderId = encBtn.dataset.labEncoder as EncoderId;
      render();
      return;
    }

    const horBtn = target.closest<HTMLElement>("[data-lab-horizon]");
    if (horBtn && horBtn.dataset.labHorizon) {
      horizonSeconds = Number(horBtn.dataset.labHorizon) || 5;
      render();
      return;
    }

    const lpBtn = target.closest<HTMLElement>("[data-lab-logprob-mode]");
    if (lpBtn && lpBtn.dataset.labLogprobMode) {
      logprobMode = lpBtn.dataset.labLogprobMode as "raw" | "processed";
      render();
      return;
    }

    const clkBtn = target.closest<HTMLElement>("[data-lab-clock]");
    if (clkBtn && clkBtn.dataset.labClock) {
      ablationClock = clkBtn.dataset.labClock as "realtime" | "lockstep";
      render();
      return;
    }

    const varRow = target.closest<HTMLElement>("[data-lab-variant]");
    if (varRow && varRow.dataset.labVariant) {
      ablationCandidateId = varRow.dataset.labVariant as AblationVariantId;
      render();
      return;
    }

    const resetBtn = target.closest<HTMLElement>("#lab-reset");
    if (resetBtn) {
      reset();
    }
  });

  host.addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter" && ev.key !== " ") return;
    const target = ev.target as Element | null;
    const nodeEl = target?.closest<HTMLElement | SVGElement>("[data-lab-node]");
    if (nodeEl) {
      ev.preventDefault();
      const nodeId = nodeEl.getAttribute("data-lab-node");
      if (nodeId) selectNode(nodeId);
    }
  });

  host.addEventListener("input", (ev) => {
    const target = ev.target as HTMLInputElement | null;
    if (target && target.id === "lab-latency-slider") {
      latencyMs = Number(target.value) || 55;
      updateHorizon();
    }
  });

  render();

  return {
    refresh: render,
    selectChapter,
    selectNode,
    selectScenario,
    selectRcaCase,
    reset,
    getState() {
      return {
        activeChapterId,
        selectedNodeId,
        scenarioId,
        rcaCaseId,
        hardware,
        encoderId,
        latencyMs,
        horizonSeconds,
        logprobMode,
        ablationCandidateId,
        ablationClock,
      };
    },
  };
}
