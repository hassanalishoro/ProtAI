# ProtAI — Product Context

This file is the source-of-truth context for any frontend design work on
ProtAI. The `impeccable` skill (and any AI agent helping with design)
must read this before producing visual or interaction work.

---

## What ProtAI is

A graph neural network framework for protein–ligand binding-affinity
prediction. Trained on the MISATO dataset (16,972 protein–ligand
complexes × 100 molecular-dynamics frames each, ~124 GB of structural
data) with experimental affinity labels from PDBbind.

The headline scientific claim — and what the site exists to communicate
— is that **trajectory-aware training trades out-of-distribution
generalization for in-distribution accuracy, and a multitask formulation
with an MD-energy auxiliary partially resolves this trade-off**.

The headline numbers (use them verbatim, never round):

- In-distribution test (random_logk split, n = 1,579):
  Pearson 0.331 (static crystal) → 0.405 (trajectory) → **0.414 (multitask)**
- Cross-evaluation (similarity split, n = 1,565):
  Pearson 0.311 (static) vs 0.056 (trajectory) vs 0.244 (multitask)

The site is **not a product launch**. It is a research demonstration
artifact for a final-year project (FYP-2) that doubles as a portfolio
piece and the public face of an associated research paper.

---

## Register

This project mixes both impeccable registers:

| Surface | Register | Why |
|---|---|---|
| `index.astro` (landing) | **brand** | Design IS the product. First impression matters more than affordance. |
| `architecture.astro` | **brand** | Long-form explainer; gravitas + readability over interactivity. |
| `methodology.astro` | **brand** | Same — long-form scientific explainer. |
| `results.astro` | **brand** | Data-dense but the experience IS the data; treat tables/figures as design objects. |
| `reference.astro` | **brand** | Bibliography page. |
| `demo.astro` | **product** | The interactive 3D viewer is the technical core; affordance and feedback dominate aesthetic. |

When unclear, default to **brand** — this is a research site first, an app second.

---

## Audiences (in priority order)

1. **Drug-discovery researchers and computational chemists** — must
   immediately recognize the work is methodologically serious. They
   look for: target choice (PDBbind log K, not raw MD energy), reported
   metrics (Pearson + Spearman + R² + RMSE in correct units), honest
   cross-evaluation results, link to the paper, link to the GitHub
   source. They will leave instantly if they smell marketing fluff.

2. **Recruiters / FYP reviewers / academic supervisors** — looking for
   evidence of independent execution (deployed demo, public code, written
   paper). Want to verify the project is real and the student understood
   what they built. They will read the methodology page closely.

3. **Students and curious onlookers** — looking for an accessible
   explainer of what ML for binding affinity actually does. They are the
   bonus audience; do not optimize for them at the expense of (1) and (2).

The site succeeds when audience (1) finishes the landing page believing
the work is credible enough to read further.

---

## Brand voice

**Confident but not boastful. Specific but not jargon-heavy. Honest
about limitations.**

Three voice rules:

- **Numbers carry the message, not adjectives.** "Pearson 0.41 in
  distribution, 0.06 cross-evaluation" beats "strong performance with
  identified generalization opportunities".

- **Acknowledge limitations as findings.** The OOD collapse is a feature
  of the story, not something to hide. "Trajectory training overfits
  to family-specific dynamics" is presented as an interesting discovery,
  not a confession.

- **Sentence rhythm matters.** Mix short declarative statements with
  one longer sentence per paragraph. Never two long sentences in a
  row. Never four short sentences in a row. The pacing should feel
  like a research-paper introduction — measured, deliberate, never
  breathless.

Examples of tone (use as voice references):

- ✅ "We trained on 1.7 million conformations. The model learned what
  matters. Then it forgot when shown new protein families. This paper
  is what we did about it."
- ✅ "Trajectory training improves in-distribution Pearson by 22%
  relative to the static baseline. It also collapses to near-zero
  correlation on novel protein families. A multitask formulation
  recovers most of that gap."
- ❌ "ProtAI revolutionizes drug discovery with cutting-edge AI."
- ❌ "Leveraging state-of-the-art deep learning to deliver superior
  binding affinity prediction."
- ❌ "Get started in seconds with our intuitive interface."

---

## Anti-references (what the site should NOT look or feel like)

- **No DNA double-helix backgrounds.** Cliché.
- **No stock photos** of doctors, lab coats, pills, or generic
  gloved-hand-holding-pipette shots. None.
- **No "AI for healthcare" corporate gradient** (saturated blue +
  purple radial behind a 3D rendering of nothing).
- **No Lottie animations** of cute molecules waving or generic
  data-point pulse pings.
- **No pastel pinks/purples.** This isn't a wellness app.
- **No glassmorphism for its own sake.** Use sparingly when it adds
  depth between layered information; never as a default surface treatment.
- **No skeuomorphic 3D buttons or shadows.** The aesthetic is flat with
  intentional depth, not Material Design / iOS faux-physical.
- **No marketing-speak power words.** "Revolutionary", "cutting-edge",
  "seamless", "intuitive", "next-generation", "world-class", "best-in-class".
  All banned.
- **No ROI/business-case framing.** This is a research artifact, not a
  startup pitch deck.
- **No fake social proof.** No invented testimonials, no fictional
  "trusted by" logos, no "20+ researchers" counters.

---

## Visual references (study these before designing)

**Tier 1 — match this energy:**

- **Linear.app** — typography rhythm, restrained palette with one
  accent, dense-information layouts that breathe, micro-interactions
  with intent.
- **AlphaFold (alphafold.ebi.ac.uk)** — scientific authority through
  restraint. Note how they let the protein structures be the visual.
- **Anthropic.com** — confident research-lab tone, generous whitespace,
  serif headlines that signal gravitas without preciousness.

**Tier 2 — borrow specific moves:**

- **Vercel.com** — interactive demo presentation, hover states with
  depth, the "this thing actually does what it claims" framing.
- **Resend.com** — animation timing and choreography. Their micro-
  interactions are the gold standard for SaaS without being overdone.
- **Stripe Press** — typographic hierarchy, scientific paper density
  done beautifully, monospace tabular numbers that align like ledgers.

**Tier 3 — for the demo page specifically:**

- **NGL Viewer (nglviewer.org)** — utilitarian molecular viewer; the
  interactions and panel layout are battle-tested for chemistry work.
- **PyMOL gallery (pymol.org/gallery)** — scientific visualization
  conventions. Atom coloring, surface rendering, what "good" looks like.

**Tier 4 — what NOT to be (study to avoid):**

- Generic Tailwind UI components used unmodified (every-startup look).
- "AI-first SaaS" landing pages with floating cards and gradient blobs.
- Bootstrap-era enterprise dashboards (the current frontend leans this way; the redesign should erase that DNA).

---

## Strategic principles

- **The 3D viewer is the hero technical artifact.** Wherever it
  appears (landing hero, demo page), it should feel like the thing the
  rest of the site exists to support.
- **Every number is real and traceable.** No invented metrics, no
  rounded-for-marketing values. Numbers come from the runs, the JSONs,
  the report tables. Use tabular figures everywhere.
- **Honesty is the brand.** The OOD collapse, the limitations, the
  "we deferred pretraining to future work" — these stay visible. Hiding
  them weakens the work.
- **Mobile is a real first-class target,** not an afterthought. The
  3DMol viewer must work on mobile, the tables must reflow, the
  navigation must collapse cleanly.
- **Performance is a credibility signal.** A research-grade site that
  loads slowly contradicts the "we built this carefully" message.
  Lighthouse Performance ≥ 90, LCP < 2.5s on slow 3G.
- **Accessibility is non-negotiable.** WCAG AA on contrast in both
  themes. Every interactive element keyboard-accessible. Screen reader
  labels on icon-only buttons. Reduced-motion respected end-to-end.

---

## Existing constraints (do not violate)

- **Stack is fixed:** Astro 6.x + React 19 + Tailwind 3.4 + Lucide icons
  + GSAP. Do not migrate to anything else.
- **Backend API contract is fixed.** All `/api/*` calls go to a Flask
  app on `:5000`. Request/response shapes are documented in
  `src/lib/api.ts`. Do not change them.
- **Routing is file-based via Astro.** Do not introduce a different
  router.
- **Build target is static** (`astro build`). Site must work on any
  static host (Vercel, Netlify, Cloudflare Pages, GitHub Pages).
- **No global CSS-in-JS.** Tailwind only. No styled-components, no
  emotion.
- **3DMol.js viewer logic in `src/components/demo/Viewer.tsx` works
  correctly.** Restyle the wrapper, do not break the rendering path.

---

## Pages and their job-to-be-done

| Page | Primary job | Secondary job | Required CTAs |
|---|---|---|---|
| `index.astro` | Convince audience (1) the work is credible in < 30 seconds | Drive to `/demo` | "Try the demo" + "Read the paper" |
| `demo.astro` | Let the visitor predict an affinity themselves | Make the model feel alive | None — the interaction IS the CTA |
| `architecture.astro` | Explain the SchNet + multitask architecture | Show the data pipeline | "See it run" → `/demo` |
| `methodology.astro` | Walk through dataset, splits, training procedure | Establish methodological seriousness | Link to the paper |
| `results.astro` | Display the four headline tables and five figures | Provide numbers reviewers can quote | Cite-this-work block |
| `reference.astro` | Bibliography, attribution, license | Link to MISATO/PDBbind/PyG | Link to GitHub |

---

## Things to ask the user before starting design work

If any of these are ambiguous, pause and ask:

1. Which color palette direction (Lab Specimen / Microscope Glass /
   Synchrotron — see DESIGN.md). Default: Lab Specimen.
2. Display font choice (Instrument Serif default; Source Serif 4 or
   Fraunces as alternatives).
3. Which PDB to use as the rotating hero molecule. Default: `1A1B`
   (already the demo's default).
4. Whether to redesign the wordmark or keep the existing one in
   `Logo.astro`.
5. Deployment target (affects asset paths). Likely Cloudflare Pages or
   Vercel.

---

## Repo and external links

- GitHub: `https://github.com/LastPredator/ProtAI`
- MISATO dataset: `https://zenodo.org/records/7711953`
- MISATO repo: `https://github.com/t7morgen/misato-dataset`
- PDBbind: `http://www.pdbbind-plus.org.cn/`
- PyTorch Geometric: `https://pytorch-geometric.readthedocs.io/`
- 3DMol.js: `https://3dmol.csb.pitt.edu/`

---

## Authors / institution

- Hassan Ali Shoro (22I-0561) — model architecture, evaluation, frontend
- Ibaad Ahmed Chaudhry (22I-0585) — data engineering, cloud deployment
- Abdullah Kaif Sheikh (22I-2142) — data pipeline, configuration system
- Supervisor: Mr. Shoaib Saleem Khattak
- Institution: National University of Computer and Emerging Sciences (FAST), Islamabad
- Year: 2026

End of PRODUCT.md.
