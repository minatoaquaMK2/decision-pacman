// Pure language helpers for the site: no DOM access, so they run under Vitest.

export type Lang = "zh" | "en";
export type Copy = { [key: string]: string | Copy };

export function isLang(v: unknown): v is Lang {
  return v === "zh" || v === "en";
}

/**
 * Language order: `?lang=` in the URL, then the saved choice, then English.
 */
export function pickLang(search: string, saved: string | null, _browserLangs: readonly string[] = []): Lang {
  const asked = new URLSearchParams(search).get("lang");
  if (isLang(asked)) return asked;
  if (isLang(saved)) return saved;
  return "en";
}

/** Looks up a dotted key ("hero.title"); returns the key itself when it is missing, so gaps show on the page. */
export function lookup(copy: Copy, key: string): string {
  let node: string | Copy | undefined = copy;
  for (const part of key.split(".")) {
    if (node === undefined || typeof node === "string") return key;
    node = node[part];
  }
  return typeof node === "string" ? node : key;
}

/** Fills `{name}` placeholders; unknown names stay as they are. */
export function fill(template: string, vars: Record<string, string | number> = {}): string {
  return template.replace(/\{(\w+)\}/g, (m, name: string) => (name in vars ? String(vars[name]) : m));
}

/** Every dotted key with a string value, for checking that both languages carry the same keys. */
export function flattenKeys(copy: Copy, prefix = ""): string[] {
  const out: string[] = [];
  for (const [k, v] of Object.entries(copy)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (typeof v === "string") out.push(key);
    else out.push(...flattenKeys(v, key));
  }
  return out.sort();
}
