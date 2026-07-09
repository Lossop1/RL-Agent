import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..", "src");
const appRoot = path.resolve(import.meta.dirname, "..");
const backendRoot = path.resolve(appRoot, "..", "autotuner", "locomotion_console");
const forbiddenMetaphors = [
  "贾维斯",
  "战甲",
  "Jarvis",
  "armor",
  "suit of armor",
  "比喻",
  "副驾",
  "装备",
];
const mojibakeMarkers = ["�", "锛", "鈥", "鍒", "绯", "杩", "鐨", "妯"];

const jsxEnglishText = />[A-Za-z][^<>{}]*</g;
const allowedFiles = new Set([
  path.join(root, "i18n", "zh-CN.ts"),
  path.join(root, "i18n", "format.ts"),
]);

function walk(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(full));
    else if (/\.(tsx|ts|css|md)$/.test(entry.name)) out.push(full);
  }
  return out;
}

const failures = [];
const indexHtml = fs.readFileSync(path.join(appRoot, "index.html"), "utf8");
if (!/<html\s+lang="zh-CN"/.test(indexHtml)) {
  failures.push("index.html: missing lang=\"zh-CN\"");
}
if (!/<meta\s+charset="UTF-8"\s*\/?>/.test(indexHtml)) {
  failures.push("index.html: missing UTF-8 charset declaration");
}

for (const file of walk(root)) {
  const text = fs.readFileSync(file, "utf8");
  const rel = path.relative(path.resolve(import.meta.dirname, ".."), file);

  for (const marker of mojibakeMarkers) {
    if (text.includes(marker)) {
      failures.push(`${rel}: contains possible mojibake marker "${marker}"`);
    }
  }

  for (const word of forbiddenMetaphors) {
    if (text.includes(word)) {
      failures.push(`${rel}: contains forbidden metaphor token "${word}"`);
    }
  }

  if (!allowedFiles.has(file) && file.endsWith(".tsx")) {
    const matches = text.match(jsxEnglishText) || [];
    for (const match of matches) {
      failures.push(`${rel}: JSX English text ${match}`);
    }
  }
}

for (const file of walk(backendRoot)) {
  const text = fs.readFileSync(file, "utf8");
  const rel = path.relative(appRoot, file);

  for (const marker of mojibakeMarkers) {
    if (text.includes(marker)) {
      failures.push(`${rel}: contains possible mojibake marker "${marker}"`);
    }
  }

  for (const word of forbiddenMetaphors) {
    if (text.includes(word)) {
      failures.push(`${rel}: contains forbidden metaphor token "${word}"`);
    }
  }
}

if (failures.length) {
  console.error("i18n check failed:");
  for (const item of failures) console.error(`- ${item}`);
  process.exit(1);
}

console.log("i18n check passed");
