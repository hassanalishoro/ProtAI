# ProtAI — Design System

This file is the source-of-truth design system for the ProtAI frontend.
The `impeccable` skill (and any AI agent helping with design) must read
this alongside PRODUCT.md before producing visual work.

PRODUCT.md = strategy and constraints.
DESIGN.md  = concrete tokens, components, motion.

---

## 1. Color palette

### Three palette directions

The site supports a dark theme (default) and a light theme (toggle).
Build all three palettes below as Tailwind theme extensions; ship
**Lab Specimen** as the default dark, **Microscope Glass** as the light
counterpart, and **Synchrotron** as an opt-in alternative dark theme
controllable via a CSS data attribute (`<html data-theme="synchrotron">`).

#### Option A — Lab Specimen (default dark)

Calm, near-black with a hint of cool. The site's resting state.

```css
--color-bg          : #0A0E1A;   /* page background */
--color-surface-1   : #11162A;   /* cards, raised surfaces */
--color-surface-2   : #161D36;   /* nested surfaces, hover state */
--color-surface-3   : #1F2540;   /* borders, dividers */

--color-text-1      : #E6E8EE;   /* primary body text */
--color-text-2      : #B6BCC9;   /* secondary text, captions */
--color-text-3      : #8B91A8;   /* tertiary text, placeholders */

--color-accent      : #FF6B4A;   /* binding "hot", coral, primary CTA */
--color-accent-soft : #FF6B4A1F; /* 12% accent for backgrounds */
--color-cool        : #4ABFD0;   /* binding "cold", teal */

--color-positive    : #7DD3A8;   /* success, good metric */
--color-warning     : #F5C26B;   /* warning */
--color-negative    : #E97070;   /* error, bad metric */

--color-data-protein: #4ABFD0;   /* protein atoms in viewers */
--color-data-ligand : #F5C26B;   /* ligand atoms in viewers */
--color-data-pocket : #7DD3A8;   /* binding-site highlight */
```

#### Option B — Microscope Glass (default light)

Off-white, clinical, high-readability.

```css
--color-bg          : #FAFBFC;
--color-surface-1   : #FFFFFF;
--color-surface-2   : #F4F5F8;
--color-surface-3   : #E5E8EE;

--color-text-1      : #0B1426;
--color-text-2      : #404758;
--color-text-3      : #707788;

--color-accent      : #0066FF;   /* clinical blue */
--color-accent-soft : #0066FF14;
--color-cool        : #0EA5C4;

--color-positive    : #16A34A;
--color-warning     : #D97706;
--color-negative    : #DC2626;

--color-data-protein: #0EA5C4;
--color-data-ligand : #D97706;
--color-data-pocket : #16A34A;
```

#### Option C — Synchrotron (optional alternative dark)

High-energy, chromatic. Use only on the landing hero or a dedicated
"poster" page. Not the default.

```css
--color-bg          : #0A0014;
--color-bg-gradient : linear-gradient(180deg, #0A0014 0%, #1A0033 100%);
--color-surface-1   : #1A0033;
--color-surface-2   : #2A0048;

--color-text-1      : #F0E4FF;
--color-text-2      : #B89DD9;

--color-accent      : #EC4899;   /* magenta — binding sites */
--color-cool        : #10B981;   /* emerald — protein */
--color-data-ligand : #F59E0B;   /* amber — ligand */
```

### Contrast requirements

Every text-on-surface combination must meet **WCAG AA**:
- Body text (≤ 18px regular / ≤ 14px bold): 4.5:1 minimum
- Large text (≥ 18px regular / ≥ 14px bold): 3:1 minimum

Verify with a contrast checker BEFORE shipping. The defaults above all
pass; if you tweak any color, recheck.

---

## 2. Typography

### Type families

Three fonts. Self-host via `@fontsource/*` packages; do NOT rely on
Google Fonts CDN.

```ts
// tailwind.config.ts -> theme.extend.fontFamily
fontFamily: {
  display: ['"Instrument Serif"', 'Source Serif 4', 'Georgia', 'serif'],
  sans:    ['"Inter Variable"', 'Inter', 'system-ui', 'sans-serif'],
  mono:    ['"JetBrains Mono Variable"', 'JetBrains Mono',
            'ui-monospace', 'SFMono-Regular', 'monospace'],
}
```

- **Display (serif)** — used for h1, h2, headline numbers in hero blocks,
  pull quotes. Italic variant for emphasis on numerical results.
- **Sans (Inter)** — body text, UI labels, navigation, buttons.
  Variable weight; use 400 for body, 500 for UI controls, 600 for h3+
  and chips.
- **Mono (JetBrains)** — every measured number. Wrap in
  `class="font-mono tabular-nums"` so digits don't shift width during
  animations or scrubber updates.

### Type scale

Modular scale (1.250 ratio), rounded to clean pixel values:

| Token | Size (rem / px) | Use |
|---|---|---|
| `text-xs` | 0.75 / 12 | mono captions, footnotes |
| `text-sm` | 0.875 / 14 | secondary text, table cells |
| `text-base` | 1.0 / 16 | body text default |
| `text-lg` | 1.125 / 18 | lead paragraphs |
| `text-xl` | 1.375 / 22 | h4, large UI labels |
| `text-2xl` | 1.75 / 28 | h3 |
| `text-3xl` | 2.25 / 36 | h2 |
| `text-4xl` | 3.0 / 48 | h1 in content pages |
| `text-5xl` | 4.0 / 64 | landing-hero secondary |
| `text-6xl` | 5.0 / 80 | landing-hero primary |

### Line-height + letter-spacing

```ts
// tailwind.config.ts -> theme.extend.lineHeight / letterSpacing
lineHeight: {
  tight: '1.1',     // display headlines
  snug:  '1.3',     // h2 / h3
  base:  '1.55',    // body
  relaxed: '1.7',   // long-form prose (methodology, architecture pages)
}
letterSpacing: {
  tighter: '-0.02em',  // display headlines
  tight:   '-0.01em',  // h2 / h3
  normal:  '0',
  wide:    '0.02em',   // small caps, mono labels
}
```

### Numerical display rules

- **Always** wrap measured numbers in `font-mono tabular-nums`.
- Pearson, Spearman, R² to **3 decimal places**: `0.405`, never `0.4`
  or `0.40`.
- RMSE / MAE to **2 decimal places** when in −log K units, **1 decimal**
  in kcal/mol: `1.66`, `50.3`.
- Sample sizes use thousand separators with non-breaking space:
  `1,579` complexes (in HTML use `&thinsp;` for narrow spacing).
- Percentages to **1 decimal**: `+22.4%`, never `+22%`.

---

## 3. Spacing and layout

### Spacing scale

Tailwind's default is fine. Use multiples of `0.25rem` (4px). Common values:

| Token | px | Use |
|---|---|---|
| `1` | 4 | tight icon spacing |
| `2` | 8 | inside chips |
| `3` | 12 | inside small buttons |
| `4` | 16 | default gap, inside cards |
| `6` | 24 | card padding, grid gutter |
| `8` | 32 | section internal spacing |
| `12` | 48 | between subsections |
| `16` | 64 | between sections (mobile) |
| `24` | 96 | between sections (tablet) |
| `30` | 120 | between sections (desktop) |

### Container widths

```ts
// tailwind.config.ts -> theme.extend.maxWidth
maxWidth: {
  prose:   '65ch',   // long-form text
  content: '1200px', // standard cards, tables
  wide:    '1440px', // viewer panels, full-width data
}
```

Always horizontally center with `mx-auto px-6 md:px-8 lg:px-12`.

### Grid

12-column grid on desktop, 1-column on mobile. Explicit breakpoints:

```ts
screens: {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1440px',
}
```

Mobile-first. Every layout starts as a stack and breaks into a grid at
`md` or `lg`. Do not try to design desktop-first.

---

## 4. Elevation and depth

### Border + shadow scale

```css
--border-1 : 1px solid var(--color-surface-3);
--border-2 : 1px solid var(--color-text-3);

--shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.20);
--shadow-md: 0 4px 8px -2px rgba(0, 0, 0, 0.30),
             0 2px 4px -1px rgba(0, 0, 0, 0.20);
--shadow-lg: 0 12px 24px -6px rgba(0, 0, 0, 0.40),
             0 4px 8px -2px rgba(0, 0, 0, 0.25);
```

In **dark mode**, prefer **borders over shadows** for delineation —
shadows on dark surfaces look muddy. Use `border-1` for resting card
state, `border-2` (or accent border) for hover.

In **light mode**, shadows work normally. Use `shadow-sm` for resting,
`shadow-md` for hover.

### Glass / blur usage

Reserved for **two specific cases only**:
1. Sticky header that overlays scrolling content (subtle backdrop-blur).
2. Modal/sheet overlays.

Never as a default surface treatment. No floating gradient cards.

---

## 5. Component spec sheet

Build these in `src/components/ui/` as a tight library that pages compose
from. All components: TypeScript, props typed, no inline styles, no
arbitrary Tailwind values (everything via the theme).

### `Button.tsx`

```ts
type ButtonProps = {
  variant?: 'primary' | 'secondary' | 'ghost' | 'link';
  size?: 'sm' | 'md' | 'lg';
  icon?: ReactNode;     // optional leading icon
  trailingIcon?: ReactNode;
  loading?: boolean;
  disabled?: boolean;
  asChild?: boolean;    // render-as-child for asChild link semantics
  children: ReactNode;
}
```

- **primary** — bg=`accent`, text=`bg`, hover lifts brightness by 8%.
- **secondary** — bg=`surface-2`, border=`surface-3`, text=`text-1`.
- **ghost** — transparent bg, hover bg=`surface-2`.
- **link** — text=`accent`, underline on hover.
- All variants: focus ring 2px `accent` with 2px offset; transitions 150ms.

### `Card.tsx`

```ts
type CardProps = {
  variant?: 'default' | 'accent' | 'outlined';
  interactive?: boolean;  // adds hover lift + border intensify
  padding?: 'sm' | 'md' | 'lg';  // 16 / 24 / 32
  children: ReactNode;
}
```

### `StatBlock.tsx`

```ts
type StatBlockProps = {
  value: string | number;
  label: string;
  caption?: string;       // small mono text below
  delta?: { value: number; positive: boolean };  // +0.074 chip
  size?: 'sm' | 'md' | 'lg';
  animateOnView?: boolean;  // count-up animation
}
```

Rendered structure:
```
[ tiny uppercase label ]
  HUGE_MONO_NUMBER  [optional delta chip]
  caption text
```

### `MetricBar.tsx`

```ts
type MetricBarProps = {
  value: number;       // -log K value
  max: number;
  label: string;       // "Predicted" or "True"
  color?: 'accent' | 'cool';  // accent for predicted, cool for true
}
```

Horizontal bar, animated width on mount, label + value to the right.

### `Table.tsx`

Booktabs-style. Top + bottom thick rules, thin midrules between header
and body, no vertical lines, alternating row tint, mono right-aligned
numbers, hover row highlight.

```ts
type TableProps = {
  headers: { label: string; align?: 'left' | 'right' | 'center' }[];
  rows: { cells: ReactNode[]; highlight?: boolean }[];
  caption?: string;
  footer?: ReactNode;
}
```

### `Chip.tsx`

Pill, 12px text, 4px y-padding, 12px x-padding. Variants: `default`,
`accent`, `success`, `warning`, `negative`. Optional leading icon.

### `Section.tsx`

Wrapper with vertical padding + max-width + intersection-observer reveal.

```ts
type SectionProps = {
  width?: 'prose' | 'content' | 'wide';
  paddingY?: 'sm' | 'md' | 'lg';  // 64 / 96 / 120
  reveal?: boolean;  // default true; fades in on scroll
  id?: string;
  children: ReactNode;
}
```

### `CodeBlock.tsx`

Monospace, syntax-themed, copy button in top-right that flashes
"Copied" for 1.2s on click. Use `shikiji` for highlighting if available;
fall back to a manual token-color CSS theme.

### `Tooltip.tsx`

Shows on hover after 300ms delay. Dark surface, 12px text, max-width
240px. Arrow pointing at the trigger. Esc to dismiss.

### `Sheet.tsx`

Mobile-first slide-up drawer for the demo page's stats panel. Backdrop
blur, swipe-down-to-dismiss, focus trap.

### `Nav.astro`

Sticky top, backdrop blur, 64px tall. Logo on left, page links centered,
GitHub icon + theme toggle on right. On scroll past 80px, gain a 1px
bottom border. Mobile: hamburger collapses page links into a sheet.

### `Footer.astro`

Three columns on desktop, stacked on mobile:
1. Wordmark + 1-line tagline
2. Page navigation
3. References (GitHub, MISATO, PDBbind, PyG)
Below: institution + year + supervisor + license note in `text-xs text-3`.

---

## 6. Motion language

### Timing tokens

```css
--duration-instant : 100ms;
--duration-fast    : 150ms;
--duration-base    : 250ms;
--duration-slow    : 400ms;
--duration-slower  : 600ms;

--ease-out-expo    : cubic-bezier(0.16, 1, 0.3, 1);    /* default for entrances */
--ease-out-quint   : cubic-bezier(0.22, 1, 0.36, 1);   /* default for movement */
--ease-in-out-quint: cubic-bezier(0.83, 0, 0.17, 1);   /* default for state changes */
```

Defaults to use unless a specific case warrants otherwise:
- **Hover state changes**: 150ms ease-out-quint
- **Card lifts on hover**: 200ms ease-out-quint, transform translateY(-2px)
- **Section reveals on scroll**: 600ms ease-out-expo, translateY(24px) + opacity 0→1
- **Modal/sheet open**: 300ms ease-out-expo
- **Number count-ups**: 800ms ease-out-quint
- **3DMol viewer mode transitions**: tween via 3DMol's built-in animation, 400ms

### Motion rules (non-negotiable)

1. **No bounces, no overshoots** unless explicitly part of a feedback
   gesture (e.g. a brief scale 1 → 1.05 → 1 on prediction confirmation).
2. **Respect `prefers-reduced-motion`** at the source — wrap every
   animation in `media (prefers-reduced-motion: no-preference)` or
   `useReducedMotion` hook for React.
3. **Stagger siblings** by 60–80ms when revealing a list, never more.
4. **Single-fire scroll reveals** — once an element enters viewport, it
   stays revealed; do not re-animate when scrolling back.
5. **Page transitions use Astro's View Transitions API** — crossfade
   between routes, never feel like a page reload.
6. **Loading states are skeletons, not spinners** — except for explicit
   "computing" actions (the demo's predict button can show a small
   spinner).

### Hero molecule animation

The landing-page rotating molecule is the site's signature motion:

- Continuous Y-axis rotation, **30 seconds per full revolution** (slow
  enough to feel meditative, not distracting)
- Coloring: by adaptability gradient (`color-cool` → `color-accent`)
- On hover: rotation pauses, atom scale gently increases (1.05) — let
  the user inspect
- On `prefers-reduced-motion`: rotation disabled, static render

---

## 7. Iconography

- **Lucide React** for all UI icons. Strict 16/20/24 px sizes.
- **Stroke width 1.75** for all icons (Lucide default is 2; reduce
  for a slightly more refined feel).
- No emoji as icons. Anywhere.
- Custom SVG only for the wordmark and the architecture diagram.

---

## 8. Imagery and visualizations

- The five report figures (`fig_training_curves.png`, `fig_loss_curves.png`,
  `fig_cross_eval.png`, `fig_predictions.png`, `fig_metric_panel.png`)
  live in `public/figures/`. They were generated from real run data;
  do not regenerate them in the frontend.
- For the landing page hero, embed a 3DMol.js viewer with PDB 1A1B
  (matches the demo's default).
- Any inline data viz (sparklines, mini-charts) should use plain SVG
  or Recharts — not Chart.js, not D3 (overkill here).

---

## 9. Accessibility checklist (gating ship)

- [ ] WCAG AA contrast verified in both light and dark themes
- [ ] All interactive elements keyboard-accessible
- [ ] Visible focus rings on every interactive element (NOT
      `outline: none` without replacement)
- [ ] Skip-to-content link as the first focusable element
- [ ] All images have `alt` attributes (decorative ones use `alt=""`)
- [ ] Icon-only buttons have `aria-label`
- [ ] Heading hierarchy is sequential (no h2 → h4 jumps)
- [ ] Forms have visible labels (no placeholder-only labels)
- [ ] Error states are announced to screen readers via `aria-live`
- [ ] `prefers-reduced-motion` disables every non-essential animation

---

## 10. Performance budget

- **Lighthouse Performance**: ≥ 90 on landing page
- **LCP**: < 2.5s on simulated slow 3G
- **CLS**: < 0.1 on every page
- **Total transferred bundle (landing)**: < 250 KB (excluding the
  3DMol.js library which is ~700 KB itself; load it lazily on demo + hero)
- **Images**: WebP/AVIF with width/height attributes; PNG fallback only
  for the report figures
- **Fonts**: subset to Latin only; preload `Inter Variable` and
  `Instrument Serif` regular weight

---

## 11. File / directory conventions

```
src/
├── components/
│   ├── ui/             # generic primitives (Button, Card, Table, ...)
│   ├── nav/            # Nav, Footer, ThemeToggle
│   ├── home/           # landing-page-specific components (Hero, etc)
│   ├── demo/           # demo-page-specific (existing files)
│   └── viz/            # shared visualization components (StatBlock, MetricBar)
├── layouts/
│   └── Default.astro   # single layout, used by all pages
├── lib/
│   ├── api.ts          # backend API wrapper (do not modify the contract)
│   ├── animations.ts   # reusable GSAP / IntersectionObserver helpers
│   ├── cn.ts           # className merging (clsx + tailwind-merge)
│   └── format.ts       # number formatters (always tabular)
├── pages/
│   ├── index.astro
│   ├── demo.astro
│   ├── architecture.astro
│   ├── methodology.astro
│   ├── results.astro
│   └── reference.astro
├── styles/
│   ├── global.css      # CSS variables, base reset, font-face
│   └── tokens.css      # theme tokens generated from tailwind config
└── content/
    ├── findings.ts     # the four headline findings as data
    └── references.ts   # bibliography
```

---

## 12. Tailwind config requirements

```ts
// tailwind.config.ts must export theme tokens for:
//  - colors           (all from section 1, mapped to semantic names)
//  - fontFamily       (display, sans, mono — section 2)
//  - fontSize         (full type scale — section 2)
//  - lineHeight       (tight, snug, base, relaxed)
//  - letterSpacing    (tighter, tight, normal, wide)
//  - maxWidth         (prose, content, wide)
//  - screens          (sm/md/lg/xl/2xl)
//  - boxShadow        (sm, md, lg from section 4)
//  - borderRadius     (sm 4px, md 8px, lg 12px, xl 16px)
//  - transitionTimingFunction (ease-out-expo, ease-out-quint, etc)
//  - transitionDuration (matches motion tokens)
//
// Do NOT use arbitrary Tailwind values (e.g. `text-[#FF6B4A]`) anywhere
// outside the global CSS / tokens layer. Every visual decision must
// resolve to a named token.
```

---

End of DESIGN.md.
