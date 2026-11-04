import raw from "./openai-evaluation.json";

type Provenance = "fresh" | "historical";
type Provider = "openai" | "jev";
type Translator = (key: string, vars?: Record<string, string | number>) => string;
type NullableMetrics<K extends string> = Record<K, number | null>;

interface PacmanRow extends NullableMetrics<"games" | "pellets" | "latencyMs" | "failures"> {
  model: string;
  modelId: string;
  provenance: Provenance;
}
interface RcaRow extends NullableMetrics<"syntheticCorrect" | "realCorrect" | "syntheticLatencyMs" | "realLatencyMs" | "syntheticFailures" | "realFailures"> {
  model: string;
  modelId: string;
  provenance: Provenance;
}
export interface UsageRow extends NullableMetrics<"requests" | "inputTokens" | "cachedTokens" | "cacheWriteTokens" | "outputTokens" | "reasoningTokens" | "costUsd" | "unknownUsageRequests"> {
  task: string;
  model: string;
  modelId: string;
  provider: Provider;
}
interface EvaluationReport {
  date: string;
  models: string[];
  pacman: PacmanRow[];
  rca: RcaRow[];
  syntheticN: number;
  realN: number;
  majorityCorrect: number;
  usage: UsageRow[];
  pricing: typeof raw.pricing;
  sources: typeof raw.sources;
}

function provenance(value: string): Provenance {
  if (value === "fresh" || value === "historical") return value;
  throw new Error(`Unknown evaluation provenance: ${value}`);
}

function provider(value: string): Provider {
  if (value === "openai" || value === "jev") return value;
  throw new Error(`Unknown evaluation provider: ${value}`);
}

export const evaluation: EvaluationReport = {
  ...raw,
  pacman: raw.pacman.map((row) => ({ ...row, provenance: provenance(row.provenance) })),
  rca: raw.rca.map((row) => ({ ...row, provenance: provenance(row.provenance) })),
  usage: raw.usage.map((row) => ({ ...row, provider: provider(row.provider) })),
};

function escape(value: string): string {
  return value.replace(/[&<>"']/g, (character) => {
    switch (character) {
      case "&": return "&amp;";
      case "<": return "&lt;";
      case ">": return "&gt;";
      case '"': return "&quot;";
      default: return "&#39;";
    }
  });
}

export function accuracyText(correct: number, total: number): string {
  return `${(correct / total * 100).toFixed(1)}% (${correct}/${total})`;
}

export function summarizeSpend(rows: readonly UsageRow[]) {
  return {
    knownUsd: rows.reduce((sum, row) => sum + (row.costUsd ?? 0), 0),
    pendingTasks: rows.filter((row) => row.costUsd === null).length,
    unknownRequests: rows.reduce((sum, row) => sum + (row.unknownUsageRequests ?? 0), 0),
    requests: rows.reduce((sum, row) => sum + (row.requests ?? 0), 0),
  };
}

function money(value: number): string {
  return `$${value.toFixed(6)}`;
}

export function evaluationMarkup(report: EvaluationReport, t: Translator): { comparison: string; usage: string } {
  const text = (key: string, vars?: Record<string, string | number>) => escape(t(`evaluation.${key}`, vars));
  const metric = (value: number | null, source: Provenance = "fresh") => value === null
    ? text(source === "fresh" ? "pending" : "unrecorded")
    : escape(value.toLocaleString("en-US", { maximumFractionDigits: 1 }));
  const source = (value: Provenance) => `<span class="evaluation-source is-${value}">${text(value)}</span>`;
  const freshRca = report.rca.filter((row) => row.provenance === "fresh");
  const spendStatus = (value: ReturnType<typeof summarizeSpend>) => text(
    value.pendingTasks > 0 ? "cost_pending" : value.unknownRequests > 0 ? "cost_incomplete" : "cost_complete",
    { requests: value.requests, unknown: value.unknownRequests },
  );
  const taskCost = (row: Pick<PacmanRow, "modelId" | "provenance">, tasks: readonly string[]) => {
    if (row.provenance === "historical") return text("unrecorded");
    const rows = report.usage.filter((usage) => usage.modelId === row.modelId && tasks.includes(usage.task));
    if (tasks.some((task) => !rows.some((usage) => usage.task === task))) return text("unrecorded");
    const spend = summarizeSpend(rows);
    if (spend.pendingTasks > 0 || rows.some((usage) => usage.unknownUsageRequests === null)) return text("pending");
    return `${money(spend.knownUsd)}${spend.unknownRequests > 0 ? `<small>${spendStatus(spend)}</small>` : ""}`;
  };
  const providers = [...new Set(report.usage.map((row) => row.provider))];
  const providerNames: Record<Provider, string> = { openai: "OpenAI", jev: "TypeSafe Jev" };
  const headings = (keys: string[]) => `<tr>${keys.map((key) => `<th scope="col">${text(key)}</th>`).join("")}</tr>`;
  return {
    comparison: `
    <p class="kicker">${text("kicker")} · <time datetime="${escape(report.date)}">${escape(report.date)}</time></p>
    <h2 class="heading">${text("title")}</h2>
    <p class="lead">${text("lead")}</p>
    <p class="evaluation-caveat">${text("responses_protocol")}</p>
    <article class="card evaluation-card">
      <h3 class="subheading">${text("pacman_title")}</h3>
      <p>${text("pacman_protocol")}</p>
      <div class="evaluation-table-scroll" tabindex="0" role="region" aria-label="${text("pacman_title")}">
        <table class="lab-table evaluation-table" id="evaluation-pacman-table">
          <thead>${headings(["model", "provenance", "games", "pellets", "latency", "failures", "comparison_cost"])}</thead>
          <tbody>${report.pacman.map((row) => `<tr class="is-${row.provenance}"><th scope="row">${escape(row.model)}</th><td>${source(row.provenance)}</td><td>${metric(row.games, row.provenance)}</td><td>${metric(row.pellets, row.provenance)}</td><td>${metric(row.latencyMs, row.provenance)}</td><td>${metric(row.failures, row.provenance)}</td><td>${taskCost(row, ["pacman"])}</td></tr>`).join("")}</tbody>
        </table>
      </div>
      <p class="evaluation-caveat">${text("pacman_cost_scope")}</p>
      <p class="evaluation-caveat">${text("historical_note")}</p>
      <a href="${escape(report.sources.pacman)}">${text("pacman_source")}</a>
    </article>
    <article class="card evaluation-card">
      <h3 class="subheading">${text("rca_title")}</h3>
      <p>${text("rca_protocol", { synthetic: report.syntheticN, real: report.realN })}</p>
      <div class="evaluation-table-scroll" tabindex="0" role="region" aria-label="${text("rca_title")}">
        <table class="lab-table evaluation-table" id="evaluation-rca-table">
          <thead>${headings(["model", "provenance", "synthetic", "real", "synthetic_latency", "real_latency", "comparison_cost"])}</thead>
          <tbody>${report.rca.map((row) => `<tr class="is-${row.provenance}"><th scope="row">${escape(row.model)}</th><td>${source(row.provenance)}</td><td>${row.syntheticCorrect === null ? text("pending") : accuracyText(row.syntheticCorrect, report.syntheticN)}</td><td>${row.realCorrect === null ? text("pending") : accuracyText(row.realCorrect, report.realN)}</td><td>${metric(row.syntheticLatencyMs, row.provenance)}</td><td>${metric(row.realLatencyMs, row.provenance)}</td><td>${taskCost(row, ["rca_synthetic", "rca_real"])}</td></tr>`).join("")}</tbody>
        </table>
      </div>
      <p class="evaluation-caveat">${text("rca_cost_scope", { synthetic: report.syntheticN, real: report.realN })}</p>
      <p class="evaluation-caveat">${text("majority_note", { accuracy: accuracyText(report.majorityCorrect, report.realN) })}</p>
      ${freshRca.map((row) => `<p class="evaluation-caveat">${text("rca_failures", { model: row.model, synthetic: row.syntheticFailures ?? t("evaluation.pending"), real: row.realFailures ?? t("evaluation.pending") })}</p>`).join("")}
      <p class="evaluation-caveat">${text("rca_limits")}</p>
      <a href="${escape(report.sources.rca)}">${text("rca_source")}</a>
    </article>`,
    usage: `<article class="card evaluation-card">
      <h2 class="subheading" id="evaluation-usage-title">${text("usage_title")}</h2>
      <p>${text("cost_basis")}</p>
      <div class="evaluation-table-scroll" tabindex="0" role="region" aria-label="${text("provider_cost_title")}">
        <table class="lab-table evaluation-table" id="evaluation-provider-cost-table">
          <caption>${text("provider_cost_title")}</caption>
          <thead>${headings(["provider", "requests", "cost"])}</thead>
          <tbody>${providers.map((name) => {
            const providerSpend = summarizeSpend(report.usage.filter((row) => row.provider === name));
            return `<tr data-provider="${name}"><th scope="row">${providerNames[name]}</th><td>${metric(providerSpend.requests)}</td><td>${money(providerSpend.knownUsd)}<small>${spendStatus(providerSpend)}</small></td></tr>`;
          }).join("")}</tbody>
        </table>
      </div>
      <div class="evaluation-table-scroll" tabindex="0" role="region" aria-label="${text("usage_title")}">
        <table class="lab-table evaluation-table" id="evaluation-usage-table">
          <thead>${headings(["model", "task", "requests", "input", "cache_read", "cache_write", "output", "reasoning", "cost"])}</thead>
          <tbody>${report.usage.map((row) => `<tr data-model="${escape(row.modelId)}"><th scope="row">${escape(row.model)}</th><td>${text(`tasks.${row.task}`)}</td><td>${metric(row.requests)}</td><td>${metric(row.inputTokens)}</td><td>${row.provider === "jev" ? text("not_applicable") : metric(row.cachedTokens)}</td><td>${row.provider === "jev" ? text("not_applicable") : metric(row.cacheWriteTokens)}</td><td>${metric(row.outputTokens)}</td><td>${row.reasoningTokens === null ? text("not_applicable") : metric(row.reasoningTokens)}</td><td>${row.costUsd === null ? text("pending") : money(row.costUsd)}</td></tr>`).join("")}</tbody>
        </table>
      </div>
      <div class="evaluation-table-scroll" tabindex="0" role="region" aria-label="${text("rates_title")}">
        <table class="lab-table evaluation-table" id="evaluation-pricing-table">
          <caption>${text("rates_title")}</caption>
          <thead>${headings(["model", "uncached_rate", "cache_read", "cache_write", "output", "pricing_source"])}</thead>
          <tbody>${report.pricing.map((row) => `<tr><th scope="row">${escape(row.model)}</th><td>$${row.uncached}</td><td>${row.cached === null ? text("not_applicable") : `$${row.cached}`}</td><td>${row.cacheWrite === null ? text("not_applicable") : `$${row.cacheWrite}`}</td><td>$${row.output}</td><td><a href="${escape(row.source)}">${text("official_pricing")}</a></td></tr>`).join("")}</tbody>
        </table>
      </div>
      <p class="evaluation-formula">${text("formula")}</p>
      <p class="evaluation-formula">${text("jev_formula")}</p>
      <p class="evaluation-caveat">${text("usage_note")}</p>
      <div class="evaluation-links"><a href="${escape(report.sources.promptCaching)}">${text("cache_source")}</a></div>
    </article>`,
  };
}

function rcaBars(report: EvaluationReport, split: "synthetic" | "real", t: Translator): string {
  const total = split === "synthetic" ? report.syntheticN : report.realN;
  const rows = report.rca.map((row) => {
    const correct = split === "synthetic" ? row.syntheticCorrect : row.realCorrect;
    const latency = split === "synthetic" ? row.syntheticLatencyMs : row.realLatencyMs;
    const summary = correct === null ? escape(t("evaluation.pending")) : accuracyText(correct, total);
    const width = correct === null ? 0 : correct / total * 100;
    return `<div class="lab-bar-row"><div class="lab-bar-meta"><span>${escape(row.model)} <small class="evaluation-source is-${row.provenance}">${escape(t(`evaluation.${row.provenance}`))}</small></span><b class="lab-mono">${summary}${latency === null ? "" : ` · ${latency.toFixed(0)} ms`}</b></div><div class="lab-bar-track"><div class="lab-bar-fill ${row.provenance === "fresh" ? "is-strong" : "is-plain"}" style="width:${width}%"></div></div></div>`;
  });
  if (split === "real") {
    rows.push(`<div class="lab-bar-row"><div class="lab-bar-meta"><span>${escape(t("evaluation.majority_label"))}</span><b class="lab-mono">${accuracyText(report.majorityCorrect, total)}</b></div><div class="lab-bar-track"><div class="lab-bar-fill is-muted" style="width:${report.majorityCorrect / total * 100}%"></div></div></div>`);
  }
  return rows.join("");
}

export function renderEvaluation(t: Translator): void {
  const markup = evaluationMarkup(evaluation, t);
  const report = document.getElementById("evaluation-report");
  if (report) report.innerHTML = markup.comparison;
  const usage = document.getElementById("evaluation-usage-report");
  if (usage) usage.innerHTML = markup.usage;
  for (const split of ["synthetic", "real"] as const) {
    const host = document.getElementById(`rca-${split}-comparison`);
    if (host) host.innerHTML = rcaBars(evaluation, split, t);
  }
  const caveat = document.getElementById("rca-comparison-caveat");
  if (caveat) caveat.textContent = t("evaluation.rca_bar_note", { date: evaluation.date });
}
