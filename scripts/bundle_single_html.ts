import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = fileURLToPath(new URL("..", import.meta.url));
const distDir = path.join(rootDir, "dist-site");
const indexHtmlPath = path.join(distDir, "index.html");

if (!fs.existsSync(indexHtmlPath)) {
  throw new Error(`Missing ${indexHtmlPath}. Run "npm run build:site" first.`);
}

let html = fs.readFileSync(indexHtmlPath, "utf8");

const cssMatch = html.match(/<link\s+rel="stylesheet"[^>]*href="\/decision-pacman\/(assets\/[^"]+\.css)"[^>]*>/);
if (!cssMatch) {
  throw new Error("Could not find stylesheet link in dist-site/index.html");
}
const cssRelPath = cssMatch[1];
const cssContent = fs.readFileSync(path.join(distDir, cssRelPath), "utf8");
html = html.replace(cssMatch[0], () => `<style>\n${cssContent}\n</style>`);

const jsMatch = html.match(/<script\s+type="module"[^>]*src="\/decision-pacman\/(assets\/[^"]+\.js)"[^>]*><\/script>/);
if (!jsMatch) {
  throw new Error("Could not find module script in dist-site/index.html");
}
const jsRelPath = jsMatch[1];
let jsContent = fs.readFileSync(path.join(distDir, jsRelPath), "utf8");

const mediaDir = path.join(distDir, "media");
let inlinedMediaCount = 0;
if (fs.existsSync(mediaDir)) {
  for (const file of fs.readdirSync(mediaDir)) {
    const fullPath = path.join(mediaDir, file);
    if (!fs.statSync(fullPath).isFile()) continue;
    const ext = path.extname(file).toLowerCase();
    const mime =
      ext === ".mp4"
        ? "video/mp4"
        : ext === ".jpg" || ext === ".jpeg"
          ? "image/jpeg"
          : ext === ".png"
            ? "image/png"
            : "application/octet-stream";
    const b64 = fs.readFileSync(fullPath).toString("base64");
    const dataUri = `data:${mime};base64,${b64}`;
    const targetToken = `media/${file}`;
    if (jsContent.includes(targetToken)) {
      jsContent = jsContent.replaceAll(targetToken, () => dataUri);
      inlinedMediaCount++;
    }
  }
}

const safeJs = jsContent.replace(/<\/script/gi, "<\\/script");

html = html.replace(jsMatch[0], () => "").replace(/^[ \t]+$/gm, "");
html = html.replace("</body>", () => `<script type="module">\n${safeJs}\n</script>\n</body>`);

const outRootPath = path.join(rootDir, "decision-pacman-single.html");
const outDistPath = path.join(distDir, "decision-pacman-single.html");
fs.writeFileSync(outRootPath, html, "utf8");
fs.writeFileSync(outDistPath, html, "utf8");

const sizeKb = (Buffer.byteLength(html, "utf8") / 1024).toFixed(1);
console.log(`Bundled single-file HTML (${sizeKb} KB, ${inlinedMediaCount} media files inlined):`);
console.log(`  - ${outRootPath}`);
console.log(`  - ${outDistPath}`);
