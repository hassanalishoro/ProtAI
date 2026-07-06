import { defineConfig } from "astro/config";

/**
 * The frontend ships the design-handoff prototype as a static SPA.
 *
 * Layout:
 *   public/index.html         - HTML shell, theme bootstrap, font loading
 *   public/styles.css         - design tokens + component CSS
 *   public/scripts/*.{js,jsx} - React + Babel-in-browser app
 *   public/figures/*.png      - real run-derived figures (training_curves,
 *                               loss_curves, predictions, cross_eval,
 *                               metric_panel) used on the results page
 *
 * The Astro framework is here only as a dev server with a Vite proxy that
 * forwards /api/* to the Flask backend on :5000. There are no .astro
 * pages — `npm run build` simply copies `public/` into `dist/` for static
 * deployment.
 *
 * Why the prototype-as-static approach:
 *   - The handoff IS pixel-perfect React + JSX; porting it to Astro
 *     pages with embedded React islands wasted hours fighting parser
 *     edge cases for no functional gain.
 *   - In-browser Babel adds ~150 kB and a one-time JIT pass at startup,
 *     which is acceptable for a research demo (loads in < 1s on a
 *     normal connection).
 *   - For a production-grade pre-compile, swap the in-browser Babel
 *     scripts for a Vite-bundled build later.
 */
export default defineConfig({
  vite: {
    server: {
      proxy: {
        // Single regex catches every backend route without enumerating each.
        "^/api(/.*)?$": {
          target: "http://localhost:5000",
          changeOrigin: true,
        },
      },
    },
  },
  prefetch: true,
});
