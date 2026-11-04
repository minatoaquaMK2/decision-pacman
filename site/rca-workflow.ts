type Translator = (key: string, vars?: Record<string, string | number>) => string;

export interface CostScenario {
  classificationShare: number;
  replacementPercent: number;
  extraPreparation: number;
}

export function scenarioCost(scenario: CostScenario) {
  const unchanged = 100 - scenario.classificationShare;
  const classifier = scenario.classificationShare * scenario.replacementPercent / 100;
  const total = unchanged + classifier + scenario.extraPreparation;
  return { unchanged, classifier, preparation: scenario.extraPreparation, total, saving: 100 - total };
}

const scenario: CostScenario = { classificationShare: 10, replacementPercent: 10, extraPreparation: 5 };
const controls = [
  { key: "classificationShare", max: 100 },
  { key: "replacementPercent", max: 100 },
  { key: "extraPreparation", max: 50 },
] satisfies { key: keyof CostScenario; max: number }[];
const stages = ["evidence", "prepare", "classify", "investigate", "resolve"];

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

function number(value: number): string {
  return value.toLocaleString("en-US", { maximumFractionDigits: 1 });
}

export function workflowCostMarkup(value: CostScenario, t: Translator): string {
  const text = (key: string, vars?: Record<string, string | number>) => escape(t(`workflow.${key}`, vars));
  const cost = scenarioCost(value);
  const scale = Math.max(100, cost.total);
  const segment = (kind: string, amount: number) => `<span class="workflow-segment is-${kind}" style="width:${amount / scale * 100}%"></span>`;
  const result = cost.saving > 0 ? "saved" : cost.saving < 0 ? "extra" : "same";
  return `
    <div class="workflow-cost-row"><div><span>${text("baseline")}</span><b>${text("units", { n: 100 })}</b></div><div class="workflow-cost-track" aria-hidden="true">${segment("unchanged", cost.unchanged)}${segment("original", value.classificationShare)}</div></div>
    <div class="workflow-cost-row"><div><span>${text("replacement")}</span><b>${text("units", { n: number(cost.total) })}</b></div><div class="workflow-cost-track" aria-hidden="true">${segment("unchanged", cost.unchanged)}${segment("classifier", cost.classifier)}${segment("preparation", cost.preparation)}</div></div>
    <ul class="workflow-cost-legend">
      <li><i class="is-unchanged" aria-hidden="true"></i>${text("unchanged", { n: number(cost.unchanged) })}</li>
      <li><i class="is-original" aria-hidden="true"></i>${text("original", { n: number(value.classificationShare) })}</li>
      <li><i class="is-classifier" aria-hidden="true"></i>${text("classifier", { n: number(cost.classifier) })}</li>
      <li><i class="is-preparation" aria-hidden="true"></i>${text("preparation", { n: number(cost.preparation) })}</li>
    </ul>
    <p class="workflow-cost-result is-${result}" aria-live="polite" id="workflow-cost-result">${text(result, { n: number(Math.abs(cost.saving)) })}</p>
    <p class="workflow-equation">100 − ${number(value.classificationShare)} + ${number(value.classificationShare)} × ${number(value.replacementPercent)}% + ${number(value.extraPreparation)} = <b>${number(cost.total)}</b></p>`;
}

export function workflowMarkup(t: Translator, value: CostScenario): string {
  const text = (key: string, vars?: Record<string, string | number>) => escape(t(`workflow.${key}`, vars));
  return `
    <p class="kicker">${text("kicker")}</p>
    <h2 class="heading" id="workflow-title">${text("title")}</h2>
    <p class="lead">${text("lead")}</p>
    <figure class="workflow-diagram">
      <ol class="workflow-stages">${stages.map((stage, index) => `<li class="workflow-stage is-${stage}"><span class="workflow-step">0${index + 1} · ${text(`stages.${stage}.owner`)}</span><h3>${text(`stages.${stage}.title`)}</h3><p>${text(`stages.${stage}.body`)}</p>${stage === "classify" ? `<span class="workflow-scope">${text("measured_scope")}</span>` : ""}</li>`).join("")}</ol>
      <div class="workflow-evidence"><span aria-hidden="true">↳</span><p>${text("evidence_path")}</p></div>
      <figcaption>${text("diagram_note")}</figcaption>
    </figure>
    <div class="card workflow-calculator">
      <div class="workflow-calculator-head"><div><p class="kicker">${text("illustration")}</p><h3 class="subheading">${text("calculator_title")}</h3></div><span class="workflow-assumption">${text("baseline_units")}</span></div>
      <p>${text("calculator_lead")}</p>
      <div class="workflow-calculator-grid">
        <div class="workflow-controls">${controls.map(({ key, max }) => `<div class="workflow-control"><label for="workflow-${key}">${text(`controls.${key}.label`)}<output for="workflow-${key}" id="workflow-${key}-value">${value[key]}%</output></label><input id="workflow-${key}" type="range" min="0" max="${max}" step="1" value="${value[key]}" aria-describedby="workflow-${key}-hint"><p id="workflow-${key}-hint">${text(`controls.${key}.hint`)}</p></div>`).join("")}</div>
        <div id="workflow-cost-bars">${workflowCostMarkup(value, t)}</div>
      </div>
      <p class="workflow-limit">${text("limit")}</p>
      <p class="evaluation-caveat">${text("accounting")}</p>
    </div>
    <div class="workflow-notes">
      <article><h3>${text("scope_title")}</h3><p>${text("scope_body")}</p><a href="https://developers.openai.com/api/docs/guides/decisions">${text("decisions_source")}</a></article>
      <article><h3>${text("context_title")}</h3><p>${text("context_body")}</p><p>${text("context_limits")}</p><a href="https://docs.typesafe.ai/models">${text("context_source")}</a></article>
    </div>`;
}

export function renderWorkflow(t: Translator): void {
  const host = document.getElementById("rca-workflow-root");
  if (!host) return;
  host.innerHTML = workflowMarkup(t, scenario);
  for (const { key } of controls) {
    const input = host.querySelector(`#workflow-${key}`);
    if (!(input instanceof HTMLInputElement)) continue;
    input.addEventListener("input", () => {
      scenario[key] = input.valueAsNumber;
      const output = host.querySelector(`#workflow-${key}-value`);
      if (output) output.textContent = `${scenario[key]}%`;
      const bars = host.querySelector("#workflow-cost-bars");
      if (bars) bars.innerHTML = workflowCostMarkup(scenario, t);
    });
  }
}
