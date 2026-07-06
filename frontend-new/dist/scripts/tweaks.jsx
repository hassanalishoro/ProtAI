/* ============================================================
   Tweaks — 3 expressive controls
   1. Palette direction (3 themes)
   2. Display voice (font character + tracking)
   3. Density (editorial vs compact)
   ============================================================ */

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "palette": "lab",
  "voice": "serif",
  "density": "editorial"
}/*EDITMODE-END*/;

// Merge defaults with whatever the bootstrap script already applied
// from localStorage — keeps the panel state in sync with first-paint.
function resolveInitial() {
  const root = document.documentElement;
  return {
    palette: root.getAttribute("data-theme")   || TWEAK_DEFAULTS.palette,
    voice:   root.getAttribute("data-voice")   || TWEAK_DEFAULTS.voice,
    density: root.getAttribute("data-density") || TWEAK_DEFAULTS.density,
  };
}

const PALETTES = {
  lab:         { label: "Lab Specimen",     sub: "default · dark, calm near-black",       swatch: ["#0A0E1A", "#FF6B4A", "#4ABFD0"] },
  microscope:  { label: "Microscope Glass", sub: "clinical light, blue accent",           swatch: ["#FAFBFC", "#0066FF", "#0EA5C4"] },
  synchrotron: { label: "Synchrotron",      sub: "high-energy poster, magenta + emerald", swatch: ["#0A0014", "#EC4899", "#10B981"] },
};

const VOICES = {
  serif:    { label: "Editorial",  sub: "Instrument Serif · slanted italics for emphasis" },
  sober:    { label: "Sober",      sub: "Source Serif 4 · less editorial swing" },
  clinical: { label: "Clinical",   sub: "Geist sans · headline becomes a label" },
};

const DENSITIES = {
  editorial: { label: "Editorial", sub: "research-paper rhythm · 120 px between sections" },
  compact:   { label: "Compact",   sub: "product dashboard · 72 px between sections" },
};

function ProtAITweaks() {
  const [t, setTweak] = window.useTweaks(resolveInitial());

  // Apply to DOM
  useEffect(() => {
    document.documentElement.setAttribute("data-theme",   t.palette);
    document.documentElement.setAttribute("data-voice",   t.voice);
    document.documentElement.setAttribute("data-density", t.density);
    try {
      localStorage.setItem("protai-palette", t.palette);
      localStorage.setItem("protai-voice",   t.voice);
      localStorage.setItem("protai-density", t.density);
    } catch {}
  }, [t.palette, t.voice, t.density]);

  // If the nav's ThemeToggle changes data-theme externally, mirror it here so
  // the panel stays in sync without a reload.
  useEffect(() => {
    const root = document.documentElement;
    const obs = new MutationObserver(() => {
      const p = root.getAttribute("data-theme");
      if (p && p !== t.palette) setTweak("palette", p);
    });
    obs.observe(root, { attributes: true, attributeFilter: ["data-theme"] });
    return () => obs.disconnect();
  }, [t.palette, setTweak]);

  const { TweaksPanel, TweakSection } = window;

  return (
    <TweaksPanel title="Tweaks">
      <TweakSection label="Palette" />
      <PaletteCards value={t.palette} onChange={(v) => setTweak("palette", v)}/>

      <TweakSection label="Display voice" />
      <VoiceCards value={t.voice} onChange={(v) => setTweak("voice", v)}/>

      <TweakSection label="Density" />
      <DensityCards value={t.density} onChange={(v) => setTweak("density", v)}/>
    </TweaksPanel>
  );
}

/* ---------- Custom expressive control: palette cards with live swatches ---------- */
function PaletteCards({ value, onChange }) {
  return (
    <div style={{ display: "grid", gap: 6 }}>
      {Object.entries(PALETTES).map(([id, p]) => {
        const active = id === value;
        return (
          <button
            key={id}
            onClick={() => onChange(id)}
            style={{
              display: "grid",
              gridTemplateColumns: "auto 1fr",
              gap: 10,
              alignItems: "center",
              padding: "8px 10px",
              borderRadius: 8,
              border: active ? "1px solid rgba(41,38,27,.55)" : "1px solid rgba(41,38,27,.12)",
              background: active ? "rgba(0,0,0,.04)" : "transparent",
              cursor: "pointer",
              textAlign: "left",
              font: "inherit",
              color: "inherit",
            }}
            aria-pressed={active}
          >
            <span style={{
              display: "inline-flex",
              width: 38, height: 22, borderRadius: 5, overflow: "hidden",
              boxShadow: "0 0 0 1px rgba(0,0,0,.08)",
            }}>
              {p.swatch.map((c, i) => (
                <span key={i} style={{ flex: 1, background: c }}/>
              ))}
            </span>
            <span style={{ display: "grid", gap: 1, minWidth: 0 }}>
              <span style={{ fontWeight: 500, fontSize: 11.5 }}>{p.label}</span>
              <span style={{ color: "rgba(41,38,27,.55)", fontSize: 10.5, lineHeight: 1.3 }}>{p.sub}</span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

/* ---------- Voice cards (show a "Aa" specimen) ---------- */
function VoiceCards({ value, onChange }) {
  const specs = {
    serif:    { fam: "'Instrument Serif', Georgia, serif", italic: true,  weight: 400 },
    sober:    { fam: "'Source Serif 4', Georgia, serif",   italic: false, weight: 500 },
    clinical: { fam: "'Geist', 'Inter Tight', system-ui",  italic: false, weight: 600 },
  };
  return (
    <div style={{ display: "grid", gap: 6 }}>
      {Object.entries(VOICES).map(([id, v]) => {
        const active = id === value;
        const s = specs[id];
        return (
          <button
            key={id}
            onClick={() => onChange(id)}
            style={{
              display: "grid",
              gridTemplateColumns: "44px 1fr",
              gap: 10,
              alignItems: "center",
              padding: "8px 10px",
              borderRadius: 8,
              border: active ? "1px solid rgba(41,38,27,.55)" : "1px solid rgba(41,38,27,.12)",
              background: active ? "rgba(0,0,0,.04)" : "transparent",
              cursor: "pointer",
              textAlign: "left",
              font: "inherit",
              color: "inherit",
            }}
            aria-pressed={active}
          >
            <span style={{
              fontFamily: s.fam,
              fontStyle: s.italic ? "italic" : "normal",
              fontWeight: s.weight,
              fontSize: 24,
              lineHeight: 1,
              letterSpacing: id === "clinical" ? "-0.03em" : "-0.015em",
              textAlign: "center",
              color: "rgba(41,38,27,.85)",
            }}>Aa</span>
            <span style={{ display: "grid", gap: 1, minWidth: 0 }}>
              <span style={{ fontWeight: 500, fontSize: 11.5 }}>{v.label}</span>
              <span style={{ color: "rgba(41,38,27,.55)", fontSize: 10.5, lineHeight: 1.3 }}>{v.sub}</span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

/* ---------- Density cards with mini layout preview ---------- */
function DensityCards({ value, onChange }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
      {Object.entries(DENSITIES).map(([id, d]) => {
        const active = id === value;
        const compact = id === "compact";
        return (
          <button
            key={id}
            onClick={() => onChange(id)}
            style={{
              padding: "8px 10px",
              borderRadius: 8,
              border: active ? "1px solid rgba(41,38,27,.55)" : "1px solid rgba(41,38,27,.12)",
              background: active ? "rgba(0,0,0,.04)" : "transparent",
              cursor: "pointer",
              textAlign: "left",
              font: "inherit",
              color: "inherit",
              display: "grid",
              gap: 6,
            }}
            aria-pressed={active}
          >
            <span style={{ display: "flex", flexDirection: "column", gap: compact ? 2 : 5, alignItems: "stretch" }}>
              <span style={{ height: compact ? 3 : 4, background: "rgba(41,38,27,.55)", width: "45%", borderRadius: 1 }}/>
              <span style={{ height: compact ? 2 : 3, background: "rgba(41,38,27,.25)", width: "75%", borderRadius: 1 }}/>
              <span style={{ height: compact ? 2 : 3, background: "rgba(41,38,27,.25)", width: "60%", borderRadius: 1 }}/>
            </span>
            <span style={{ fontWeight: 500, fontSize: 11.5 }}>{d.label}</span>
            <span style={{ color: "rgba(41,38,27,.55)", fontSize: 10.5, lineHeight: 1.3 }}>{d.sub}</span>
          </button>
        );
      })}
    </div>
  );
}

window.ProtAITweaks = ProtAITweaks;
