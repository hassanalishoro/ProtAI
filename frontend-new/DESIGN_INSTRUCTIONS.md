# ProtAI Frontend — Design Workflow

A step-by-step guide for redesigning the ProtAI frontend using a
**stack of design skills** (`impeccable`, `emil-design-eng`,
`high-end-visual-design`, etc.). This file is for the human (you), not
the agent.

---

## Files in this directory and what each does

| File | Audience | What it does |
|---|---|---|
| `PRODUCT.md` | The skills / agent | Strategy, audience, voice, anti-references |
| `DESIGN.md` | The skills / agent | Concrete tokens — colors, typography, components, motion |
| `DESIGN_INSTRUCTIONS.md` (this file) | You (the human) | Step-by-step workflow + skill stack |

The skills auto-load `PRODUCT.md` and `DESIGN.md` whenever they run in
this directory. You don't need to paste them — just keep them up to
date.

---

## The skill stack (read this once, then reference per phase)

You have **eight relevant design skills** installed. They're not
redundant — each nails a specific aspect. The right approach is
**composition**: load 2–3 skills per phase based on what that phase
needs.

### Always-on (load every session)

| Skill | Why |
|---|---|
| `redesign-existing-projects` | ProtAI is an existing site. This skill audits current state, identifies generic AI patterns, and removes them without breaking functionality. Non-negotiable starting point. |
| `full-output-enforcement` | Forces complete file output instead of `// ... rest of code` placeholders. Without this, you'll get half-finished files. |

### The driver

| Skill | Why |
|---|---|
| `impeccable` | The main workflow skill. Has sub-commands (`craft`, `shape`, `audit`, `live`, `teach`, `document`). Anthropic-derived. Use as the primary execution engine. |

### Specialists (load for specific phases)

| Skill | Best for | Phase |
|---|---|---|
| `design-taste-frontend` | Metric-based rules, hardware acceleration, balanced design engineering | 1, 2 |
| `emil-design-eng` | Polish, micro-interactions, invisible details (animations, transitions, hover states) | 4, 6 |
| `high-end-visual-design` | Awwwards-tier ambition — cinematic depth, rhythm, fluid motion | 3 |
| `gpt-taste` | GSAP scroll triggers, AIDA page structure, editorial typography | 3 |
| `minimalist-ui` | Editorial / document-style interfaces, warm monochrome, bento grids | 5 |

### Optional / utility

| Skill | When to use |
|---|---|
| `brandkit` | Only if redesigning the wordmark or building a brand-guidelines page |
| `image-to-code` | Only if you want to generate visual mockups before coding |
| `imagegen-frontend-web` | Only if you want section-by-section design references as images |

### Skip these for ProtAI

| Skill | Why skip |
|---|---|
| `industrial-brutalist-ui` | Wrong aesthetic for biotech/research |
| `stitch-design-taste` | Generates DESIGN.md for Google Stitch tool — we already have DESIGN.md |
| `imagegen-frontend-mobile` | We're not building a mobile app |

### How to invoke a skill stack in Claude Code

The cleanest pattern:

> "Use the `redesign-existing-projects`, `full-output-enforcement`, and
> `impeccable` skills together. Read PRODUCT.md and DESIGN.md first.
> Then [task description]."

Claude Code will load all three skills. Their guidelines compose: the
"don't truncate" rule from `full-output-enforcement` is enforced while
the "remove generic AI patterns" rule from `redesign-existing-projects`
guides what to output, and `impeccable` provides the workflow
structure.

---

## Phase 0 — Pre-flight (do before any design work)

Five things to verify before invoking any skill on any page.

### 0.1 Confirm all eight skills installed

```powershell
ls .agents/skills/
```

You should see at minimum: `impeccable`, `redesign-existing-projects`,
`full-output-enforcement`, `emil-design-eng`, `high-end-visual-design`,
`design-taste-frontend`, `minimalist-ui`, `gpt-taste`. If any are
missing, re-run the install commands from the earlier conversation.

Verify `impeccable` works:
```powershell
cd U:/FYP/ProtAI/frontend-new
npx impeccable --help
```

### 0.2 Make six design decisions upfront

Open `PRODUCT.md` section "Things to ask the user before starting design
work" and answer all five inline (edit the file):

1. **Color palette direction** — recommended: Lab Specimen (default
   dark) + Microscope Glass (default light). Pick.
2. **Display font** — recommended: Instrument Serif. Pick.
3. **Hero molecule PDB** — recommended: 1A1B. Pick.
4. **Wordmark** — recommended: keep current `Logo.astro` for now,
   redesign in a separate session if time permits.
5. **Deployment target** — recommended: Cloudflare Pages (free, fast,
   static-friendly). Pick.

Write your answers into PRODUCT.md so the skills pick them up
automatically; don't keep them in your head.

### 0.3 Verify the dev server still works

Before you let anything change the code, prove the existing site builds
and runs cleanly:

```powershell
cd U:/FYP/ProtAI/frontend-new
npm install
npm run dev
```

Open `http://localhost:4321` and click through every page. If anything
is broken before the redesign, fix it first (or at minimum note it so
you can tell the difference between pre-existing breakage and
redesign breakage).

### 0.4 Check the backend wires up

The demo page needs the Flask backend on `:5000`. Start it in a second
terminal:

```powershell
cd U:/FYP/ProtAI
py -3.11 backend/app.py
```

Confirm the demo page can load 1A1B and run a prediction. **The
redesign must not break this.**

### 0.5 Snapshot the current site (so you can compare)

```powershell
mkdir before
# Take screenshots of every page at 1920x1080
# Save them into before/
```

You will thank yourself later when the design review asks "what
actually changed".

---

## Phase 1 — Tokens + chrome (1–2 sessions)

**Skill stack for this phase:**
- `redesign-existing-projects` (always-on)
- `full-output-enforcement` (always-on)
- `impeccable` (driver)
- `design-taste-frontend` (metric-based rules)

### 1.1 Audit the existing site first

Before changing anything, get a baseline read:

> "Use `redesign-existing-projects` and `impeccable` (audit
> sub-command). Audit every page in `src/pages/` against PRODUCT.md
> and DESIGN.md. List the generic AI patterns to remove and the gaps
> against the new design system. Output as a checklist per page,
> nothing else."

This produces a list you can verify against later. Save it as
`before/AUDIT.md`.

### 1.2 Establish design tokens

```powershell
cd U:/FYP/ProtAI/frontend-new
npx impeccable document
```

This sub-command builds out the design tokens — should produce or
update `tailwind.config.ts` with all the color / typography / spacing
tokens defined in DESIGN.md. After it runs:

- Open `tailwind.config.ts` and confirm: colors match Section 1 of
  DESIGN.md, fonts match Section 2, container widths match Section 3.
- Open `src/styles/global.css` and confirm: CSS variables are exported
  so non-Tailwind contexts (the 3DMol viewer wrapper) can read them.
- Verify the dev server still builds (`npm run dev`).

### 1.3 Redesign Nav and Footer

Hand the agent a focused task with the full skill stack:

> "Use `redesign-existing-projects`, `full-output-enforcement`,
> `impeccable`, and `design-taste-frontend`. Read PRODUCT.md and
> DESIGN.md. Redesign Nav.astro and Footer.astro per DESIGN.md
> section 5. Sticky nav with backdrop blur on scroll, three-column
> footer with the references and team info from PRODUCT.md. Output
> the complete files — no placeholders."

Verify: open the dev server, every page has the new nav/footer chrome,
the theme toggle works, the GitHub link points at LastPredator/ProtAI,
no horizontal scroll on mobile.

### 1.4 Refresh the Default layout

> "Same skill stack. Update layouts/Default.astro to wire in the new
> typography and color tokens. Add a skip-to-content link as the first
> focusable element. Set up the page meta + OpenGraph defaults."

Verify: every page inherits the new typography on h1/h2/p/links.

**Stop here.** Pages will look unstyled but the foundation is in place.
Take screenshots of the unstyled state to track progress.

---

## Phase 2 — Component library (1 session)

**Skill stack:**
- `full-output-enforcement` (always-on — critical for component library, you DO NOT want truncated component files)
- `impeccable` (driver)
- `design-taste-frontend` (component architecture rules)

> "Use `full-output-enforcement`, `impeccable`, and
> `design-taste-frontend`. Read PRODUCT.md and DESIGN.md. Build the
> component library in `src/components/ui/` per DESIGN.md section 5.
> Build: Button, Card, StatBlock, Chip, Table, MetricBar, Section,
> CodeBlock, Tooltip, Sheet. Each component must be TypeScript with
> explicit props interface, no inline styles, no arbitrary Tailwind
> values, and include a JSDoc usage example. Add a dev-only route at
> `/_dev/components` that renders one example of each variant for
> visual QA. Output complete files — no placeholders."

Verify: visit `http://localhost:4321/_dev/components`, every component
variant renders correctly, focus rings are visible, hover states work,
mobile layout doesn't break anything.

---

## Phase 3 — Landing page (the canary, 1–2 sessions)

This is the page that sets the tone for everything else. Get it right
before doing the rest.

**Skill stack:**
- `redesign-existing-projects` (always-on)
- `full-output-enforcement` (always-on)
- `impeccable` (driver)
- `high-end-visual-design` (Awwwards-tier ambition — the hero IS the
  product on landing pages)
- `gpt-taste` (GSAP scroll choreography, AIDA structure, editorial
  typography)

> "Use `redesign-existing-projects`, `full-output-enforcement`,
> `impeccable`, `high-end-visual-design`, and `gpt-taste`. Read
> PRODUCT.md and DESIGN.md. Redesign `src/pages/index.astro` per
> PRODUCT.md section 'Pages and their job-to-be-done' and the landing
> page spec.
>
> Six sections: hero with rotating 1A1B molecule (3DMol.js, 30s/rev,
> respects prefers-reduced-motion), headline metrics strip with
> count-up animations (Pearson 0.41, R² 0.17, RMSE 1.65 from
> PRODUCT.md), three discovery cards (trajectory helps / OOD collapse
> / multitask threads needle), dataset visualization section with
> scrubbing trajectory preview, architecture one-liner with link to
> /architecture, footer CTA.
>
> Use the components from src/components/ui/. Use GSAP ScrollTriggers
> for scroll-pinning the hero, stacking the discovery cards on scroll,
> and scrubbing the dataset trajectory. Output the complete file."

Verify against the brand voice examples in PRODUCT.md. Check on
desktop, tablet, mobile. **Get user feedback here before moving to
other pages.** If the landing isn't right, the rest will inherit
its problems.

---

## Phase 4 — Demo page (1 session)

The interactive viewer is the technical hero. Restyle the chrome
without breaking the existing logic in `src/components/demo/Viewer.tsx`.

**Skill stack:**
- `redesign-existing-projects` (always-on — preserve existing
  functionality is its specialty)
- `full-output-enforcement` (always-on)
- `impeccable` (driver)
- `emil-design-eng` (UI polish, the invisible details that make the
  demo feel alive)

> "Use `redesign-existing-projects`, `full-output-enforcement`,
> `impeccable`, and `emil-design-eng`. Read PRODUCT.md (register:
> product) and DESIGN.md. Restyle `src/pages/demo.astro` and the
> wrapper chrome around `src/components/demo/Viewer.tsx`.
>
> Three-panel desktop layout (PDB selector + stats left, viewer
> center, prediction card right). Mobile: stacked, viewer first.
> Custom-styled dropdown for the quick-examples selector (the native
> one looks plain). Mode switcher as segmented control with sliding
> indicator. Frame scrubber with tick marks at 0/25/50/75/100 and
> live-updating energy label. Prediction panel with subtle pulse on
> new prediction, side-by-side bars for predicted vs actual.
>
> DO NOT modify the Viewer.tsx rendering logic or the backend API
> contract — restyle only. Output complete files."

Verify: load 1A1B, switch through the three viewing modes, scrub
through frames, run a prediction. Every interaction smoother than
before. The Flask backend should still respond correctly.

---

## Phase 5 — Content pages (1–2 sessions)

Architecture, methodology, results, reference. Apply the patterns
established on the landing page. Probably one session for two pages.

**Skill stack:**
- `redesign-existing-projects` (always-on)
- `full-output-enforcement` (always-on)
- `impeccable` (driver)
- `minimalist-ui` (editorial document-style — perfect for long-form
  scientific content)

> "Use `redesign-existing-projects`, `full-output-enforcement`,
> `impeccable`, and `minimalist-ui`. Read PRODUCT.md and DESIGN.md.
> Redesign `src/pages/architecture.astro` per the spec in PRODUCT.md.
> Pipeline diagram with animated draw-on, SchNet + multitask head
> explanation, hyperparameter table, code snippets in CodeBlock
> components. Editorial typography rhythm — generous line-height,
> measured paragraph length. Output the complete file."

Repeat for `methodology.astro`, `results.astro`, `reference.astro`
in the same skill-stack pattern. After each page, verify and screenshot.

For `results.astro` specifically, the four headline tables and five
generated figures (`public/figures/fig_*.png`) must be embedded and
look like research-paper artifacts, not stock dashboards.

---

## Phase 6 — Polish + ship-readiness (1 session)

The non-glamorous work that determines whether this looks shipped or
just demo'd.

**Skill stack:**
- `impeccable` (driver — uses `audit` sub-command extensively here)
- `emil-design-eng` (the invisible polish that separates "shipped"
  from "demoed")
- `full-output-enforcement` (always-on)

### 6.1 Page transitions

> "Use `impeccable` and `emil-design-eng`. Wire up Astro's View
> Transitions API across all pages. Crossfade between routes.
> Persistent nav (no flash on navigation). Output the complete files
> for any modified component."

### 6.2 Accessibility audit

```powershell
npx impeccable audit "All pages — verify the accessibility checklist
in DESIGN.md section 9. Report any failures. Fix anything that fails."
```

### 6.3 Performance pass

```powershell
npx impeccable audit "Run a performance pass — check Lighthouse on the
landing page, identify any unused CSS / JavaScript, verify image
optimization, confirm fonts are subset and preloaded."
```

After this, run Lighthouse manually:

```powershell
npm run build
npm run preview
# Then open Lighthouse in Chrome devtools and run on localhost:4321/
```

Targets: Performance ≥ 90, Accessibility = 100, Best Practices ≥ 95,
SEO ≥ 95.

### 6.4 Dark/light mode QA

Visit every page in both themes. Look for:
- Contrast failures
- Missing token references (raw hex values appearing)
- Component states that work in one theme but not the other (focus
  rings, hover states, disabled states)

> "Use `impeccable` (audit) and `emil-design-eng`. Dark mode and
> light mode parity check across all pages — list any visual issues
> per page and fix them."

### 6.5 Mobile QA

Resize the browser to 375px width and click through every page. The
demo page is the highest risk; the 3DMol viewer must work on mobile
Safari and Chrome.

---

## Live iteration mode (use anytime)

The skill's `live` sub-command opens a browser preview and lets you
iterate visually:

```powershell
npx impeccable live
```

Use this when you want to tweak something specific ("the prediction
panel needs more breathing room", "the headline numbers are too
loud") without writing out a full task description. The agent reads
your visual feedback and applies it.

For specific polish iterations, also invoke `emil-design-eng`:

> "Use `emil-design-eng` and the impeccable live mode. The card hover
> states feel sluggish — refine the timing and easing."

---

## Skill stack reference card (quick lookup per phase)

| Phase | Always-on | Driver | Specialists |
|---|---|---|---|
| 0 (pre-flight) | — | — | — |
| 1 (tokens + chrome) | `redesign-existing-projects`, `full-output-enforcement` | `impeccable` | `design-taste-frontend` |
| 2 (component library) | `full-output-enforcement` | `impeccable` | `design-taste-frontend` |
| 3 (landing page) | `redesign-existing-projects`, `full-output-enforcement` | `impeccable` | `high-end-visual-design`, `gpt-taste` |
| 4 (demo page) | `redesign-existing-projects`, `full-output-enforcement` | `impeccable` | `emil-design-eng` |
| 5 (content pages) | `redesign-existing-projects`, `full-output-enforcement` | `impeccable` | `minimalist-ui` |
| 6 (polish + ship) | `full-output-enforcement` | `impeccable` (audit) | `emil-design-eng` |

---

## When something goes wrong

### The agent ignores PRODUCT.md / DESIGN.md

- Confirm both files exist at `U:/FYP/ProtAI/frontend-new/`.
- Tell the agent explicitly: "Read PRODUCT.md and DESIGN.md before
  starting."
- Run `npx impeccable teach` once to re-prime the skill on the
  context.

### The agent produces generic Tailwind components

- Add `redesign-existing-projects` to the skill stack — it's
  specifically tuned to identify and remove generic AI patterns.
- Re-invoke with an explicit reference: "Per DESIGN.md section 5,
  build the Card component with the variants listed."

### The agent gives truncated files (`// ... rest of code`)

- This is exactly what `full-output-enforcement` prevents. Make sure
  it's in the skill stack.
- If it still happens, prompt: "Output the complete file. The
  `full-output-enforcement` skill explicitly bans placeholders."

### The dev server breaks after a change

- Check for missing imports / typos.
- The skill should not modify the backend API contract; if it does,
  revert with `git checkout -- src/lib/api.ts`.
- If the breakage is deep, hand the diff back: "This change broke
  X. Revert and try a different approach."

### Animations feel laggy

- Check Chrome DevTools Performance tab during the animation.
- Likely culprit: animating `width`/`height`/`top`/`left` instead of
  `transform`/`opacity`. Tell the agent: "Use `emil-design-eng` and
  audit this animation. Animate via transform/opacity only — current
  approach causes layout thrashing."

### The 3DMol viewer doesn't show up after restyle

- Check the wrapper div has explicit width/height (3DMol won't render
  into a zero-sized container).
- Check `src/components/demo/Viewer.tsx` rendering logic wasn't
  modified — only the surrounding chrome should change.

### Hero feels generic / not Awwwards-tier

- Add `high-end-visual-design` and `gpt-taste` to the stack for
  Phase 3 specifically.
- Reference specific Awwwards sites: "Match the rhythm of
  linear.app's homepage hero — slow molecule rotation, oversized
  serif headline, count-up metrics."

---

## What "done" looks like

- All six pages redesigned, consistent design language, no leftover
  generic Tailwind components
- `_dev/components` route shows the full library
- `npm run build` succeeds with zero warnings
- Lighthouse Performance ≥ 90, Accessibility = 100 on the landing page
- Demo page runs end-to-end against the backend (load PDB, switch
  modes, scrub frames, run prediction)
- Dark and light themes both look intentional
- `DESIGN.md` and `PRODUCT.md` updated to reflect any decisions made
  during the work (don't let them drift)
- Screenshots in `after/` folder for the design review

After all that — commit the whole redesign:

```powershell
cd U:/FYP/ProtAI
git add frontend-new/
git status   # review what changed
git commit -m "Redesign frontend: research-grade biotech aesthetic"
git push origin master
```

---

## Estimated total time

- Phase 0 (pre-flight): 30 min
- Phase 1 (tokens + chrome): 2–3 hours
- Phase 2 (component library): 2–3 hours
- Phase 3 (landing page): 3–4 hours (the highest-leverage phase)
- Phase 4 (demo page): 2–3 hours
- Phase 5 (content pages): 3–4 hours
- Phase 6 (polish + ship): 2–3 hours

**Total: ~14–20 hours of agent-driven work.** Spread across 2–4 days
of part-time iteration, with you reviewing screenshots between phases.

---

## Pro tip: bookmark this command for every Claude Code session

This is the canonical opener that loads the always-on skills + the
context files in one shot:

> "Read PRODUCT.md, DESIGN.md, and DESIGN_INSTRUCTIONS.md from this
> directory. Use the `redesign-existing-projects`,
> `full-output-enforcement`, and `impeccable` skills as the baseline
> stack. I will tell you which additional specialist skills to load
> per phase. Confirm you've loaded everything before I give you the
> phase task."

Paste that at the start of every session. Saves you from re-explaining
the project context.

---

End of DESIGN_INSTRUCTIONS.md.
