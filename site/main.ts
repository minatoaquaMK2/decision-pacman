import "./style.css";
import { renderEvaluation } from "./evaluation.ts";
import { renderWorkflow } from "./rca-workflow.ts";
import data from "./data.json";
import en from "./copy/en.json";
import zh from "./copy/zh.json";
import { drawChart, setupChart, type Bar } from "./chart.ts";
import { attachSource, setupPair, setupPhone } from "./clips.ts";
import { fill, lookup, pickLang, type Copy, type Lang } from "./i18n.ts";
import { buildScenarioBank, type AblationRow } from "./lab.ts";
import { mountMechanismLab } from "./lab-view.ts";
import { PlaySection } from "./play.ts";
import { setupTimers } from "./timers.ts";

const COPY: Record<Lang, Copy> = { en, zh };
const STORE_KEY = "decision-pacman.lang";

function savedLang(): string | null {
  try {
    return localStorage.getItem(STORE_KEY);
  } catch {
    return null;
  }
}

function saveLang(lang: Lang): void {
  try {
    localStorage.setItem(STORE_KEY, lang);
  } catch {
    // Remembering the choice is a convenience.
  }
}

let lang: Lang = pickLang(location.search, savedLang(), navigator.languages?.length ? navigator.languages : [navigator.language]);

const t = (key: string, vars?: Record<string, string | number>) => fill(lookup(COPY[lang], key), vars);

/** Placeholder values for copy keys; numbers live in data.json, not in the copy. */
const VARS: Record<string, Record<string, string | number>> = {
  "hero.jev_stat": data.clips.jev,
  "hero.ours_stat": data.clips.ours,
  "halo.phi_stat": data.clips.phi4_mini,
  "hero.note": { jev_avg: data.averages.jev, ours_avg: data.averages.ours },
  "paths.b_body": data.thinking,
  "paths.b_timer_fast": { s: data.timer.fast_s },
  "paths.b_timer_slow": { s: data.timer.slow_s },
};

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
}

function formatInlineCode(escaped: string): string {
  return escaped.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
}

/** Copy with placeholders: the filled-in numbers are wrapped so they can stand out. */
function fillHtml(template: string, vars: Record<string, string | number>): string {
  return formatInlineCode(escapeHtml(template)).replace(/\{(\w+)\}/g, (m, name: string) =>
    name in vars ? `<b class="num">${escapeHtml(String(vars[name]))}</b>` : m,
  );
}

const listeners: (() => void)[] = [];

function applyCopy(): void {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.title = t("meta.title");
  document.querySelector('meta[name="description"]')?.setAttribute("content", t("meta.description"));
  document.querySelector('meta[property="og:title"]')?.setAttribute("content", t("meta.title"));
  document.querySelector('meta[property="og:description"]')?.setAttribute("content", t("meta.description"));
  for (const el of document.querySelectorAll<HTMLElement>("[data-i18n]")) {
    const key = el.dataset.i18n!;
    const vars = VARS[key];
    if (vars) {
      el.innerHTML = fillHtml(lookup(COPY[lang], key), vars);
    } else {
      const text = t(key);
      if (text.includes("`") && !(el instanceof SVGElement)) {
        el.innerHTML = formatInlineCode(escapeHtml(text));
      } else {
        el.textContent = text;
      }
    }
  }
  const cta = document.getElementById("cta-button") as HTMLAnchorElement | null;
  if (cta) cta.href = lang === "zh" ? data.links.cta_zh : data.links.cta_en;
  for (const fn of listeners) fn();
}

function setLang(next: Lang): void {
  lang = next;
  saveLang(next);
  try {
    // Keep the choice in the address bar, so a copied link opens in the same language.
    const url = new URL(location.href);
    url.searchParams.set("lang", next);
    history.replaceState(null, "", url);
  } catch {
    // file:// origins in some browsers disallow replaceState with query strings.
  }
  applyCopy();
}

// ---- static wiring ---------------------------------------------------------------
document.getElementById("lang-toggle")!.addEventListener("click", () => setLang(lang === "zh" ? "en" : "zh"));

for (const a of document.querySelectorAll<HTMLAnchorElement>("a[data-link]")) {
  a.href = data.links[a.dataset.link as keyof typeof data.links];
}

const commands = document.getElementById("commands");
if (commands) {
  for (const cmd of data.commands) {
    const row = document.createElement("div");
    row.className = "cmd";
    const code = document.createElement("code");
    code.textContent = cmd;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.dataset.i18n = "footer.copy";
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(cmd);
        btn.textContent = t("footer.copied");
        setTimeout(() => (btn.textContent = t("footer.copy")), 1500);
      } catch {
        // Clipboard access can be refused; the command stays selectable.
      }
    });
    row.append(code, btn);
    commands.appendChild(row);
  }
}

// ---- clips -------------------------------------------------------------------------
const clips = data.clips as Record<string, { src: string; poster: string }>;
for (const v of document.querySelectorAll<HTMLVideoElement>("video[data-clip]")) attachSource(v, clips[v.dataset.clip!]);
for (const pair of document.querySelectorAll<HTMLElement>("[data-pair]")) setupPair(pair);

const phoneLabel = document.getElementById("phone-sound-label")!;
let phoneMuted = true;
const relabelPhone = () => {
  phoneLabel.textContent = t(phoneMuted ? "halo.unmute" : "halo.mute");
};
setupPhone(document.querySelector<HTMLVideoElement>(".phone video")!, document.getElementById("phone-sound") as HTMLButtonElement, (muted) => {
  phoneMuted = muted;
  relabelPhone();
});
listeners.push(relabelPhone);

// ---- chart and timers ----------------------------------------------------------------
const chartHost = document.getElementById("chart")!;
const redrawChart = setupChart(chartHost, () =>
  drawChart(chartHost, data.ladder as Bar[], data.band, (id) => t(`halo.chart_labels.${id}`), t("halo.chart_band")),
);
listeners.push(redrawChart);
listeners.push(() => renderEvaluation(t));
listeners.push(() => renderWorkflow(t));
setupTimers(document.getElementById("timers")!, data.timer.fast_s, data.timer.slow_s);
document.getElementById("jump-from")!.textContent = String(data.thinking.off);
document.getElementById("jump-to")!.textContent = String(data.thinking.on);

const labHost = document.getElementById("lab-root");
const lab = labHost
  ? mountMechanismLab(labHost, {
      scenarioBank: buildScenarioBank(data.seed),
      ablations: data.ablations as AblationRow[],
      repoUrl: data.links.repo,
      t,
    })
  : null;
if (lab) {
  listeners.push(() => lab.refresh());
}

for (const btn of document.querySelectorAll<HTMLElement>("[data-jump-chapter]")) {
  btn.addEventListener("click", () => {
    const ch = btn.dataset.jumpChapter;
    if (ch && lab) {
      lab.selectChapter(ch as Parameters<typeof lab.selectChapter>[0]);
      document.getElementById("lab")?.scrollIntoView({ behavior: "smooth" });
    }
  });
}

// ---- the game -------------------------------------------------------------------------
const play = new PlaySection({ seed: data.seed, seconds: data.game_seconds, opponents: data.ranking, t });
listeners.push(() => play.relabel());

applyCopy();

interface SiteTestHarness {
  readonly play: PlaySection;
  readonly lab: ReturnType<typeof mountMechanismLab> | null;
  readonly lang: Lang;
  readonly setLang: (next: Lang) => void;
}

declare global {
  interface Window {
    __site?: SiteTestHarness;
  }
}

window.__site = {
  play,
  lab,
  get lang() {
    return lang;
  },
  setLang,
};
