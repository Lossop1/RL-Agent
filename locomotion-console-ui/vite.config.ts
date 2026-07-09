import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server on :5173 (matches the backend CORS allow-list).
//
// build.outDir points at the repo-root `frontend/dist` that `docker/serve-frontend.mjs`
// serves in production. Emitting the build directly there makes the served bundle the same
// artifact the build produced — there is no separate copy step to forget, so the deployment
// can never silently serve a stale committed build.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/action": "http://127.0.0.1:8000",
      "/agent": "http://127.0.0.1:8000",
      "/chat": {
        target: "http://127.0.0.1:8000",
        ws: true,
      },
      "/config": "http://127.0.0.1:8000",
      "/definitions": "http://127.0.0.1:8000",
      "/diagnostics": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/llm": "http://127.0.0.1:8000",
      "/remote": "http://127.0.0.1:8000",
      "/run": {
        target: "http://127.0.0.1:8000",
        ws: true,
      },
      "/spec": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "../frontend/dist",
    emptyOutDir: true,
    // three.js lives in the lazily-loaded RobotViewer chunk; keep the vendor split explicit
    // so the main bundle stays small and the 3-D viewer only loads when opened.
    rollupOptions: {
      output: {
        manualChunks: {
          three: ["three", "urdf-loader"],
        },
      },
    },
  },
});
