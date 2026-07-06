/* ============================================================
   Shared UI primitives + Nav + Footer + 3DMol loader
   ============================================================ */

const { useState, useEffect, useRef, useCallback, useMemo } = React;

/* ---------- Hash router ---------- */
function useHashRoute() {
  const get = () => {
    const h = window.location.hash.replace(/^#\/?/, "");
    return h || "home";
  };
  const [route, setRoute] = useState(get);
  useEffect(() => {
    const onHash = () => {
      setRoute(get());
      window.scrollTo({ top: 0, behavior: "instant" });
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return route;
}

function navigate(route) {
  window.location.hash = "/" + route;
}

/* ---------- Reveal stagger (CSS handles the actual animation) ---------- */
function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll(".reveal");
    els.forEach((e, i) => {
      // Stagger the first ~8 elements above the fold for a gentle cascade.
      if (i < 8 && !e.style.getPropertyValue("--r-delay")) {
        e.style.setProperty("--r-delay", (i * 70) + "ms");
      }
    });
  }, []);
}

/* ---------- 3DMol loader ---------- */
function load3DMol() {
  if (window.$3Dmol) return Promise.resolve(window.$3Dmol);
  const existing = document.querySelector("script[data-threedmol]");
  if (existing) {
    return new Promise((res, rej) => {
      existing.addEventListener("load", () => res(window.$3Dmol));
      existing.addEventListener("error", rej);
    });
  }
  return new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = "https://unpkg.com/3dmol@2.4.0/build/3Dmol-min.js";
    s.async = true;
    s.dataset.threedmol = "1";
    s.onload = () => res(window.$3Dmol);
    s.onerror = rej;
    document.head.appendChild(s);
  });
}

const PDB_CACHE = new Map();
function fetchPDB(id) {
  if (PDB_CACHE.has(id)) return Promise.resolve(PDB_CACHE.get(id));
  return fetch(`https://files.rcsb.org/download/${id}.pdb`)
    .then((r) => {
      if (!r.ok) throw new Error("RCSB " + r.status);
      return r.text();
    })
    .then((txt) => {
      PDB_CACHE.set(id, txt);
      return txt;
    });
}

/* ---------- Number formatters ---------- */
function fmt3(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(3);
}
function fmt2(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(2);
}
function fmt1(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(1);
}
function fmtCount(n) {
  return n.toLocaleString("en-US");
}

/* ---------- Count-up ---------- */
function NumberCountUp({ value, decimals = 0, suffix = "", prefix = "", duration = 900 }) {
  const [v, setV] = useState(0);
  const ref = useRef(null);
  const fired = useRef(false);
  useEffect(() => {
    if (!ref.current) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { setV(value); return; }
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && !fired.current) {
          fired.current = true;
          const start = performance.now();
          const ease = (t) => 1 - Math.pow(1 - t, 4);
          const step = (now) => {
            const t = Math.min(1, (now - start) / duration);
            setV(value * ease(t));
            if (t < 1) requestAnimationFrame(step);
          };
          requestAnimationFrame(step);
        }
      });
    }, { threshold: 0.2 });
    obs.observe(ref.current);
    return () => obs.disconnect();
  }, [value, duration]);

  const display = decimals > 0 ? v.toFixed(decimals) : Math.round(v).toLocaleString("en-US");
  return <span ref={ref} className="font-mono tabular-nums">{prefix}{display}{suffix}</span>;
}

/* ---------- Logo ---------- */
function Logo({ size = "md" }) {
  return (
    <a href="#/home" className="logo" aria-label="ProtAI home">
      <span className="logo-mark">P</span>
      <span style={{ fontSize: size === "lg" ? "1.5rem" : "1.25rem", letterSpacing: "-0.01em" }}>
        ProtAI
      </span>
    </a>
  );
}

/* ---------- Theme toggle ---------- */
function ThemeToggle() {
  const [theme, setTheme] = useState(() => document.documentElement.getAttribute("data-theme") || "lab");
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("protai-palette", theme); } catch {}
  }, [theme]);
  const isDark = theme !== "microscope";
  return (
    <button
      onClick={() => setTheme(isDark ? "microscope" : "lab")}
      className="btn btn-ghost"
      style={{ padding: "8px", width: 40, height: 40 }}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      title={isDark ? "Light mode" : "Dark mode"}
    >
      {isDark ? (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="4"/>
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>
        </svg>
      ) : (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
        </svg>
      )}
    </button>
  );
}

/* ---------- Nav ---------- */
function Nav({ route }) {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 16);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  useEffect(() => { setOpen(false); }, [route]);

  const links = [
    { id: "demo",         label: "Demo" },
    { id: "architecture", label: "Architecture" },
    { id: "results",      label: "Results" },
    { id: "methodology",  label: "Methodology" },
    { id: "reference",    label: "Reference" },
  ];

  return (
    <>
      <header className={"nav " + (scrolled ? "scrolled" : "")}>
        <div className="container" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", width: "100%" }}>
          <Logo />
          <nav aria-label="Primary" className="nav-links" style={{ display: "flex", alignItems: "center", gap: 4 }}>
            {links.map((l) => (
              <a key={l.id} href={"#/" + l.id} className={"nav-link " + (route === l.id ? "active" : "")}>
                {l.label}
              </a>
            ))}
          </nav>
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <a
              href="https://github.com/LastPredator/ProtAI"
              target="_blank" rel="noopener noreferrer"
              className="btn btn-ghost"
              style={{ padding: 8, width: 40, height: 40 }}
              aria-label="ProtAI on GitHub"
            >
              <svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38v-1.34c-2.22.48-2.69-1.07-2.69-1.07-.36-.92-.89-1.17-.89-1.17-.73-.5.06-.49.06-.49.81.06 1.23.83 1.23.83.72 1.23 1.88.87 2.34.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.13 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.03 2.2-.82 2.2-.82.44 1.11.16 1.93.08 2.13.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.74.54 1.49v2.21c0 .21.15.46.55.38C13.71 14.53 16 11.54 16 8c0-4.42-3.58-8-8-8z"/>
              </svg>
            </a>
            <ThemeToggle />
            <button
              className="btn btn-ghost menu-btn"
              style={{ padding: 8, width: 40, height: 40 }}
              onClick={() => setOpen((o) => !o)}
              aria-label="Toggle menu"
              aria-expanded={open}
            >
              {open ? (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round"><path d="M6 6l12 12M6 18L18 6"/></svg>
              ) : (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>
              )}
            </button>
          </div>
        </div>
      </header>
      {open && (
        <div className="nav-mobile">
          {links.map((l) => (
            <a key={l.id} href={"#/" + l.id} className={"nav-link " + (route === l.id ? "active" : "")}>
              {l.label}
            </a>
          ))}
        </div>
      )}
    </>
  );
}

/* ---------- Footer ---------- */
function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div style={{
          display: "grid",
          gap: 48,
          gridTemplateColumns: "minmax(0, 1.4fr) repeat(3, minmax(0, 1fr))"
        }} className="footer-grid">
          <div>
            <Logo />
            <p className="text-sm text-2" style={{ marginTop: 14, maxWidth: 320 }}>
              A graph neural network framework for protein–ligand binding-affinity prediction, trained on the MISATO molecular dynamics dataset.
            </p>
          </div>
          <FooterCol title="Project" links={[
            ["Demo", "#/demo"],
            ["Architecture", "#/architecture"],
            ["Methodology", "#/methodology"],
            ["Results", "#/results"],
          ]}/>
          <FooterCol title="Data" links={[
            ["MISATO dataset", "https://zenodo.org/records/7711953"],
            ["MISATO repo", "https://github.com/t7morgen/misato-dataset"],
            ["PDBbind+", "http://www.pdbbind-plus.org.cn/"],
            ["PyTorch Geometric", "https://pytorch-geometric.readthedocs.io/"],
          ]}/>
          <FooterCol title="Code" links={[
            ["GitHub", "https://github.com/LastPredator/ProtAI"],
            ["3DMol.js", "https://3dmol.csb.pitt.edu/"],
            ["Reference", "#/reference"],
          ]}/>
        </div>
        <hr className="divider" style={{ margin: "48px 0 24px" }}/>
        <div className="text-xs text-3" style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
          <span>FAST NUCES Islamabad · FYP 2026 · supervised by Mr. Shoaib Saleem Khattak</span>
          <span className="font-mono">MIT License · v0.4.1</span>
        </div>
      </div>
      <style>{`
        @media (max-width: 768px) {
          .footer-grid { grid-template-columns: 1fr 1fr !important; }
        }
        @media (max-width: 480px) {
          .footer-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </footer>
  );
}

function FooterCol({ title, links }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-3" style={{ marginBottom: 14 }}>{title}</div>
      <ul style={{ listStyle: "none", padding: 0, display: "grid", gap: 8 }}>
        {links.map(([label, href]) => {
          const ext = href.startsWith("http");
          return (
            <li key={label}>
              <a
                href={href}
                target={ext ? "_blank" : undefined}
                rel={ext ? "noopener noreferrer" : undefined}
                className="text-sm text-2"
                style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
              >
                {label}
                {ext && (
                  <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" aria-hidden="true">
                    <path d="M3 3h6v6m0-6L3 9"/>
                  </svg>
                )}
              </a>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/* ---------- Section reveal helper ---------- */
function Section({ children, paddingY = "lg", id, className = "", reveal = true }) {
  const p = paddingY === "sm" ? "64px 0" : paddingY === "md" ? "96px 0" : "120px 0";
  return (
    <section id={id} className={(reveal ? "reveal " : "") + className} style={{ padding: p }}>
      <div className="container">{children}</div>
    </section>
  );
}

/* ---------- Stat block ---------- */
function StatBlock({ value, label, caption, delta, size = "md" }) {
  const sizeMap = {
    sm: { val: "text-2xl",  cap: "text-xs" },
    md: { val: "text-4xl",  cap: "text-sm" },
    lg: { val: "text-6xl",  cap: "text-sm" },
  };
  const s = sizeMap[size];
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-3" style={{ marginBottom: 8 }}>{label}</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
        <div className={"font-mono tabular-nums " + s.val} style={{ color: "var(--text-1)", fontWeight: 500 }}>{value}</div>
        {delta && (
          <span className={"chip"} style={{
            background: delta.positive ? "rgba(125, 211, 168, 0.12)" : "rgba(233, 112, 112, 0.12)",
            color: delta.positive ? "var(--positive)" : "var(--negative)",
            borderColor: "transparent"
          }}>
            {delta.positive ? "▲" : "▼"} {delta.value}
          </span>
        )}
      </div>
      {caption && <div className={"text-2 " + s.cap} style={{ marginTop: 8 }}>{caption}</div>}
    </div>
  );
}

/* ============================================================
   Molecule viewer — dual-mode

   Mode A (RCSB cartoon, used by the landing-page hero):
     <MoleculeViewer pdbId="1A1B" mode="adaptability" spin />
     - Fetches the PDB file directly from RCSB
     - Renders cartoon + hetero ligands as sticks
     - Supports the slow auto-rotation hero animation
     - "adaptability" mode is a fake gradient by residue index — it
       LOOKS like adaptability but is not the real per-atom values.
       (Justified for a marketing hero where atom-level fidelity is
       not the point.)

   Mode B (backend-driven, used by the demo page):
     <MoleculeViewer
        pdbId="1A1B"
        mode="adaptability"
        structure={api.getStructure(...)}
        pocket={api.getPocket(...)}
        frameCoords={api.getFrame(...).coordinates}  // optional
     />
     - Renders the EXACT structure the model sees: backend coordinates
       (post H-stripping + solvent removal), real per-atom adaptability,
       real binding-pocket residues.
     - When frameCoords is provided, atom positions tween to that frame.
       Scrub the slider and the molecule actually moves.
     - mode="element"      per-element coloring (CPK-style spheres)
       mode="adaptability" cool→hot per-atom heatmap from real values
       mode="pocket"       pocket residues green, ligand yellow,
                           rest faded to a ghost
   ============================================================ */

const ATOMIC_SYMBOLS = [
  "",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",
  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",
  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
  "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
  "Sn", "Sb", "Te", "I",  "Xe",
];
function atomicSymbol(z) {
  return ATOMIC_SYMBOLS[z] || "X";
}

/* Cool (#4ABFD0) → Hot (#FF6B4A) gradient. Used for the adaptability
   heatmap. t in [0, 1]; 0 = rigid (cool), 1 = flexible (hot). */
function lerpAdaptColor(t) {
  const r = Math.round(74  + t * (255 - 74));
  const g = Math.round(191 + t * (107 - 191));
  const b = Math.round(208 + t * (74  - 208));
  return `rgb(${r}, ${g}, ${b})`;
}

/* Build an XYZ-format string from backend structure data. 3DMol parses
   XYZ natively and assigns serial indices 1..N matching the input order,
   which lets us address atoms back by their original backend index. */
function buildXYZ(structure, frameCoords) {
  const coords =
    frameCoords && frameCoords.length === structure.coordinates.length
      ? frameCoords
      : structure.coordinates;
  const n = coords.length;
  const lines = new Array(n);
  for (let i = 0; i < n; i++) {
    const z = structure.atomic_numbers[i];
    const c = coords[i];
    lines[i] = `${atomicSymbol(z)} ${c[0].toFixed(3)} ${c[1].toFixed(3)} ${c[2].toFixed(3)}`;
  }
  return `${n}\nProtAI backend\n${lines.join("\n")}`;
}

/* Apply per-atom styling for one of the three demo modes. The viewer
   already has a model loaded; we only change styles. */
function applyDemoStyle(viewer, structure, pocket, mode) {
  // Always start from a faint default so missing per-atom rules don't
  // leave gaps; we then override per atom for the ones we want loud.
  viewer.setStyle({}, { sphere: { scale: 0.30, color: "#94A3B8" } });

  if (mode === "element") {
    viewer.setStyle({ atom: ["C"]  }, { sphere: { scale: 0.30, color: "#94A3B8" } });
    viewer.setStyle({ atom: ["N"]  }, { sphere: { scale: 0.30, color: "#3B82F6" } });
    viewer.setStyle({ atom: ["O"]  }, { sphere: { scale: 0.30, color: "#EF4444" } });
    viewer.setStyle({ atom: ["S"]  }, { sphere: { scale: 0.30, color: "#EAB308" } });
    viewer.setStyle({ atom: ["P"]  }, { sphere: { scale: 0.30, color: "#A855F7" } });
    viewer.setStyle({ atom: ["H"]  }, { sphere: { scale: 0.20, color: "#E6E8EE" } });
    viewer.setStyle({ atom: ["Cl"] }, { sphere: { scale: 0.30, color: "#7DD3A8" } });
    viewer.setStyle({ atom: ["Br"] }, { sphere: { scale: 0.32, color: "#B45309" } });
    return;
  }

  if (mode === "adaptability") {
    const a = structure && structure.adaptability;
    if (!a || a.length === 0) return; // fallback to default grey
    // Robust scaling: clip to 5th/95th percentile so a couple of crazy
    // atoms don't wash out the whole gradient.
    const sorted = [...a].sort((x, y) => x - y);
    const p5  = sorted[Math.floor(sorted.length * 0.05)] ?? 0;
    const p95 = sorted[Math.floor(sorted.length * 0.95)] ?? 1;
    const span = Math.max(p95 - p5, 1e-6);
    for (let i = 0; i < a.length; i++) {
      const t = Math.max(0, Math.min(1, (a[i] - p5) / span));
      const radius = 0.25 + t * 0.40;          // flexible = bigger sphere
      viewer.setStyle(
        { serial: i + 1 },
        { sphere: { scale: radius, color: lerpAdaptColor(t) } }
      );
    }
    return;
  }

  if (mode === "pocket") {
    if (!pocket) return; // fallback to default grey
    // 3DMol runs on WebGL and cannot read CSS custom properties — colors
    // must be literal hex / RGB / named values. The hex below intentionally
    // sits darker than the dark-theme `--data-pocket` (#7DD3A8) because
    // 3D-lit spheres look washed-out at light tints; a darker base reads
    // as a saturated mid-green once the WebGL specular highlight lands.

    // Step 1 — fade everything to a translucent ghost.
    viewer.setStyle({}, { sphere: { scale: 0.18, color: "#94A3B8", opacity: 0.10 } });
    // Step 2 — pocket residues highlighted in deep forest green.
    (pocket.protein_pocket_indices || []).forEach((i) => {
      viewer.setStyle(
        { serial: i + 1 },
        { sphere: { scale: 0.40, color: "#1B6B3F", opacity: 1.0 } }
      );
    });
    // Step 3 — ligand atoms highlighted amber.
    (pocket.ligand_indices || []).forEach((i) => {
      viewer.setStyle(
        { serial: i + 1 },
        { sphere: { scale: 0.45, color: "#F5C26B", opacity: 1.0 } }
      );
    });
  }
}

function MoleculeViewer({
  pdbId = "1A1B",
  mode = "adaptability",
  /* RCSB-cartoon mode (legacy, used by hero): */
  spin = false,
  ariaLabel,
  /* Backend-driven mode (used by demo): */
  structure = null,
  pocket = null,
  frameCoords = null,
}) {
  const hostRef = useRef(null);
  const viewerRef = useRef(null);
  const [status, setStatus] = useState("loading");

  // Mode B — backend-driven render. We re-run on every input change but
  // re-use the same viewer instance; only addModel + setStyle work runs.
  useEffect(() => {
    if (!structure) return; // mode A path takes over below
    let cancelled = false;
    (async () => {
      try {
        const Mol = await load3DMol();
        if (cancelled || !hostRef.current) return;
        if (!viewerRef.current) {
          viewerRef.current = Mol.createViewer(hostRef.current, {
            backgroundColor: "rgba(0,0,0,0)",
            antialias: true,
          });
        }
        const viewer = viewerRef.current;
        const xyz = buildXYZ(structure, frameCoords);
        // removeAllModels + addModel is fine perf-wise for ~3 k atoms;
        // setCoordinates would be marginally faster but more fragile
        // across 3DMol versions.
        viewer.removeAllModels();
        viewer.addModel(xyz, "xyz");
        applyDemoStyle(viewer, structure, pocket, mode);
        // Only zoom on the first render for this complex; subsequent
        // frame updates keep the user's chosen camera.
        if (!viewer._protaiCenteredFor || viewer._protaiCenteredFor !== pdbId) {
          viewer.zoomTo();
          viewer.zoom(1.0);
          viewer._protaiCenteredFor = pdbId;
        }
        viewer.render();
        setStatus("ready");
      } catch (err) {
        if (!cancelled) {
          console.error("[MoleculeViewer] backend render failed:", err);
          setStatus("error");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [structure, frameCoords, mode, pocket, pdbId]);

  // Mode A — RCSB-cartoon render (landing-page hero). Only runs when
  // structure is null (no backend data provided).
  useEffect(() => {
    if (structure) return;
    let cancelled = false;
    let rafId = null;
    (async () => {
      try {
        const [Mol, pdb] = await Promise.all([load3DMol(), fetchPDB(pdbId)]);
        if (cancelled || !hostRef.current) return;
        if (viewerRef.current) {
          try { viewerRef.current.clear(); } catch {}
        }
        const viewer = Mol.createViewer(hostRef.current, {
          backgroundColor: "rgba(0,0,0,0)",
          antialias: true,
        });
        viewerRef.current = viewer;
        viewer.addModel(pdb, "pdb");

        if (mode === "adaptability") {
          viewer.setStyle({}, { cartoon: { colorscheme: { prop: "resi", gradient: "rwb", min: 0, max: 200 } } });
          viewer.setStyle({ hetflag: true }, { stick: { colorscheme: "yellowCarbon" } });
        } else if (mode === "element") {
          viewer.setStyle({}, { stick: { colorscheme: "default", radius: 0.18 } });
        } else {
          viewer.setStyle({}, { cartoon: { color: "spectrum" } });
          viewer.setStyle({ hetflag: true }, { stick: { colorscheme: "yellowCarbon" } });
        }

        viewer.zoomTo();
        viewer.zoom(1.05);
        viewer.render();
        setStatus("ready");

        const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        if (spin && !reduce) {
          let paused = false;
          hostRef.current.addEventListener("mouseenter", () => (paused = true));
          hostRef.current.addEventListener("mouseleave", () => (paused = false));
          const tick = () => {
            if (!cancelled && viewerRef.current) {
              if (!paused) {
                viewer.rotate(0.4, "y");
                viewer.render();
              }
              rafId = requestAnimationFrame(tick);
            }
          };
          rafId = requestAnimationFrame(tick);
        }
      } catch (err) {
        if (!cancelled) setStatus("error");
      }
    })();
    return () => {
      cancelled = true;
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [pdbId, mode, spin, structure]);

  return (
    <div
      ref={hostRef}
      aria-label={ariaLabel || `3D structure of ${pdbId}`}
      style={{ position: "relative", width: "100%", height: "100%" }}
    >
      {status === "loading" && (
        <div
          style={{
            position: "absolute", inset: 0,
            display: "grid", placeItems: "center",
            color: "var(--text-3)",
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 12,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div className="spinner" /> loading {pdbId}…
          </div>
        </div>
      )}
      {status === "error" && (
        <div
          style={{
            position: "absolute", inset: 0,
            display: "grid", placeItems: "center",
            color: "var(--text-3)",
            fontSize: 12,
            textAlign: "center",
            padding: 16,
          }}
        >
          could not load structure
        </div>
      )}
    </div>
  );
}

/* ---------- Tag external resources for the rest of the app ---------- */
Object.assign(window, {
  useHashRoute, navigate, useReveal,
  load3DMol, fetchPDB,
  fmt3, fmt2, fmt1, fmtCount,
  NumberCountUp, Logo, ThemeToggle, Nav, Footer, Section, StatBlock, MoleculeViewer,
});
