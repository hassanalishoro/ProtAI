# ProtAI frontend

Astro + Tailwind + GSAP + 3DMol.js. Serves the public-facing site for ProtAI:
homepage, interactive demo, architecture explainer, results, methodology, and
API reference.

## Why this directory is gitignored

Until the model is fully trained on the cloud GPU, this directory is excluded
from git so `git clone` on the cloud pod stays small and fast. The pod doesn't
need any of this code to train — it only needs `protai/`, `configs/`, and
`data/processed/`.

When training is done and we're ready to publish the public site, remove the
`frontend-new/` lines from the repo's root `.gitignore`, commit this directory,
and replace the old `frontend/`.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Framework | [Astro 4](https://astro.build/) | Static-first, ships almost no JS by default, React islands where we need interactivity |
| Styling | [Tailwind CSS](https://tailwindcss.com/) + CSS variables | Utility-first; design tokens drive both light and dark themes |
| Components | Plain Tailwind (no shadcn/ui ceremony) | Same look, less setup |
| Animations | [GSAP 3](https://gsap.com/) + ScrollTrigger | Earned, physics-based, respects `prefers-reduced-motion` |
| 3D viewer | [3DMol.js](https://3dmol.csb.pitt.edu/) | Mature, works with PDB-format data |
| Icons | [Lucide React](https://lucide.dev/) | Single icon family, consistent stroke weight |
| Type | [Geist](https://vercel.com/font) (display), [Inter](https://rsms.me/inter/) (body), [JetBrains Mono](https://www.jetbrains.com/lp/mono/) (mono) | Loaded via Google Fonts, all open-licensed |

## Quick start

```bash
# 1. Install dependencies (run once)
cd frontend-new
npm install

# 2. Start the Flask backend (in a separate terminal, from repo root)
py -3.11 backend/app.py
# Backend listens on http://localhost:5000

# 3. Start the Astro dev server
npm run dev
# Frontend on http://localhost:4321
# /api/* requests are proxied to localhost:5000
```

## Build for production

```bash
npm run build
# Static files emit to frontend-new/dist/

npm run preview
# Preview the built site on http://localhost:4321
```

When deploying alongside Flask, point Flask's `static_folder` at
`frontend-new/dist/` (or copy the dist into the existing `frontend/`).

## Project structure

```
frontend-new/
├── astro.config.mjs              # Astro config: React + Tailwind + /api proxy
├── tailwind.config.ts            # Design tokens mapped to CSS variables
├── tsconfig.json                 # @/ path alias to src/
├── package.json
├── public/
│   └── favicon.svg               # The atom-bond glyph
└── src/
    ├── styles/
    │   └── globals.css           # Design tokens (light + dark) + base layer
    ├── layouts/
    │   └── Default.astro         # Nav + main slot + Footer + theme bootstrap
    ├── lib/
    │   ├── api.ts                # Type-safe wrappers for Flask backend
    │   ├── animations.ts         # Reusable GSAP timelines
    │   ├── cn.ts                 # Tailwind class name combiner
    │   └── team.ts               # Team metadata (single source of truth)
    ├── components/
    │   ├── Logo.astro            # Brand mark + wordmark
    │   ├── Nav.astro             # Sticky top nav
    │   ├── Footer.astro          # Footer with team credits
    │   ├── ThemeToggle.tsx       # Light/dark switch
    │   ├── NumberCounter.tsx     # Scroll-triggered animated counter
    │   ├── demo/
    │   │   ├── Viewer.tsx        # Main 3DMol viewer + state
    │   │   ├── ModeSwitcher.tsx  # Element/Adaptability/Pocket tabs
    │   │   ├── FrameScrubber.tsx # Trajectory play/pause + slider
    │   │   ├── PredictionPanel.tsx # Predicted vs true + confidence
    │   │   └── StatsPanel.tsx    # Pocket vs whole-protein stats
    │   └── architecture/
    │       ├── DiagramSVG.astro  # Static SVG of the model
    │       └── ForwardPass.tsx   # Animated forward-pass demo
    └── pages/
        ├── index.astro           # Homepage
        ├── demo.astro            # Interactive demo
        ├── architecture.astro    # Architecture explainer
        ├── results.astro         # Ablation table + metrics
        ├── methodology.astro     # Long-form methods + team
        └── api.astro             # HTTP API reference
```

## Design tokens

All colors and effects are CSS variables on `:root` (light) and `.dark` (dark).
Tailwind reads them via `rgb(var(--token) / <alpha-value>)`. To change a color
across the entire site, edit `src/styles/globals.css`.

Key tokens:

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | warm off-white | warm near-black | Page background |
| `--surface` | white | dark gray | Cards, panels |
| `--text` | near-black | near-white | Primary text |
| `--text-2` | mid-gray | light gray | Secondary text |
| `--accent` | teal-700 | teal-400 | Brand color, CTAs, active states |
| `--data-hot` | red | light red | Hot end of adaptability heatmap |
| `--data-cold` | blue | light blue | Cold end of adaptability heatmap |
| `--data-pocket` | green | light green | Binding pocket residues |
| `--data-ligand` | yellow | light yellow | Ligand atoms |

## Animation philosophy

- **Subtle, not splashy.** This is a scientific tool, not a SaaS landing page.
- **Functional, not decorative.** Every animation communicates a state change.
- **Physics-based easings.** `power3.out`, `expo.inOut`, `back.out(1.4)` — no linear, no bouncy.
- **`prefers-reduced-motion` is honored everywhere.** No exceptions.

## Accessibility

- 4.5:1 contrast on all text in both themes
- All interactive elements ≥44×44px tap target
- Skip-to-content link, visible on focus
- Visible focus rings (never `outline: none`)
- Color is never the only signal — heatmap and pocket views always have text labels
- 3DMol viewer state has text fallbacks (atom counts, mode legend, prediction values all readable to screen readers)

## Team

| Member | Roll No |
|---|---|
| Ibaad Ahmed Chaudhry | 22I-0585 |
| Abdullah Kaif Sheikh | 22I-2142 |
| Hassan Ali Shoro | 22I-0561 |

Supervised by **Mr. Shoaib Saleem Khattak**
Department of Computer Science, FAST NUCES Islamabad
Final Year Project, Session 2022–2026
