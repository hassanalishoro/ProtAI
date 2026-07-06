import type { Config } from "tailwindcss";

/**
 * Tailwind config for ProtAI's redesigned frontend.
 *
 * Design tokens are defined as CSS custom properties on `:root[data-theme="..."]`
 * in `src/styles/globals.css`. This config exposes them to Tailwind so utility
 * classes (`bg-bg`, `text-text-1`, `border-surface-3`, ...) compile against
 * the live theme variables and respond to runtime theme switches.
 *
 * Themes:
 *   data-theme="lab"          Lab Specimen — default dark
 *   data-theme="microscope"   Microscope Glass — default light
 *   data-theme="synchrotron"  Synchrotron — opt-in alternative dark
 */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,ts,tsx}"],
  darkMode: ["selector", '[data-theme="lab"], [data-theme="synchrotron"]'],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        "surface-1": "var(--surface-1)",
        "surface-2": "var(--surface-2)",
        "surface-3": "var(--surface-3)",
        "text-1": "var(--text-1)",
        "text-2": "var(--text-2)",
        "text-3": "var(--text-3)",
        accent: "var(--accent)",
        "accent-soft": "var(--accent-soft)",
        "accent-strong": "var(--accent-strong)",
        cool: "var(--cool)",
        positive: "var(--positive)",
        warning: "var(--warning)",
        negative: "var(--negative)",
        "data-protein": "var(--data-protein)",
        "data-ligand": "var(--data-ligand)",
        "data-pocket": "var(--data-pocket)",
      },
      fontFamily: {
        display: [
          'var(--font-display, "Instrument Serif")',
          '"Source Serif 4"',
          "Georgia",
          "serif",
        ],
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: [
          '"JetBrains Mono"',
          "ui-monospace",
          '"SFMono-Regular"',
          "Menlo",
          "monospace",
        ],
      },
      maxWidth: {
        prose: "65ch",
        content: "1200px",
        wide: "1440px",
      },
      borderRadius: {
        sm: "4px",
        md: "8px",
        lg: "12px",
        xl: "16px",
        "2xl": "20px",
      },
      transitionTimingFunction: {
        "out-expo": "cubic-bezier(0.16, 1, 0.3, 1)",
        "out-quint": "cubic-bezier(0.22, 1, 0.36, 1)",
      },
      keyframes: {
        fadeUp: {
          from: { opacity: "0", transform: "translateY(20px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        spin: {
          to: { transform: "rotate(360deg)" },
        },
      },
      animation: {
        "fade-up": "fadeUp 620ms cubic-bezier(0.16, 1, 0.3, 1) both",
      },
    },
  },
  plugins: [],
} satisfies Config;
