// Zero-dependency static file server for the built Vite frontend (frontend/dist).
// Serves hashed assets with long-lived caching and falls back to index.html for
// client-side routes (SPA). Node 18+ (global fetch / URL) — we run it on Node 22.
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { join, normalize, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const ROOT = resolve(process.env.FRONTEND_DIST || join(HERE, "..", "frontend", "dist"));
const PORT = Number(process.env.FRONTEND_PORT || 5173);
const HOST = process.env.FRONTEND_HOST || "0.0.0.0";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".urdf": "application/xml; charset=utf-8",
  ".stl": "application/octet-stream", // urdf-loader fetches meshes as ArrayBuffer
  ".dae": "model/vnd.collada+xml",
  ".glb": "model/gltf-binary",
  ".gltf": "model/gltf+json",
  ".wasm": "application/wasm",
};

const contentType = (p) => MIME[extname(p).toLowerCase()] || "application/octet-stream";

async function fileOrNull(p) {
  try {
    const s = await stat(p);
    if (s.isFile()) return p;
  } catch {
    /* not found */
  }
  return null;
}

const server = createServer(async (req, res) => {
  try {
    const urlPath = decodeURIComponent((req.url || "/").split("?")[0]);
    // Confine every request under ROOT — reject path traversal.
    const candidate = resolve(join(ROOT, normalize(urlPath)));
    const inRoot = candidate === ROOT || candidate.startsWith(ROOT + "/");

    let resolved = null;
    if (inRoot) {
      resolved =
        (await fileOrNull(candidate)) ||
        (urlPath.endsWith("/") ? await fileOrNull(join(candidate, "index.html")) : null);
    }
    // SPA fallback: unknown non-asset path -> index.html.
    const isAsset = resolved !== null;
    if (!resolved) resolved = join(ROOT, "index.html");

    const body = await readFile(resolved);
    res.writeHead(200, {
      "Content-Type": contentType(resolved),
      "Cache-Control":
        isAsset && candidate.startsWith(join(ROOT, "assets") + "/")
          ? "public, max-age=31536000, immutable"
          : "no-cache",
    });
    res.end(body);
  } catch {
    res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("500 Internal Server Error");
  }
});

server.listen(PORT, HOST, () => {
  console.log(`[frontend] serving ${ROOT} on http://${HOST}:${PORT}`);
});
