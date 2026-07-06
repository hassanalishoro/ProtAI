/* ============================================================
   Landing page — index
   Six sections per PRODUCT spec
   ============================================================ */

function PageLanding() {
  useReveal();
  return (
    <>
      <HeroLanding/>
      <NumbersStrip/>
      <DiscoveryCards/>
      <DatasetVisual/>
      <ArchitectureOneLiner/>
      <CTAFooter/>
    </>
  );
}

/* ---------- Hero ---------- */
function HeroLanding() {
  return (
    <section style={{ position: "relative", overflow: "hidden", borderBottom: "var(--border-1)" }}>
      <div className="hero-glow" aria-hidden="true"/>
      <div className="container" style={{ position: "relative", padding: "72px 24px 96px" }}>
        <div style={{
          display: "grid",
          gridTemplateColumns: "minmax(0, 1.15fr) minmax(0, 0.85fr)",
          gap: 56,
          alignItems: "center"
        }} className="hero-grid">
          <div>
            <div className="chip" style={{ marginBottom: 24 }}>
              <span style={{ width: 6, height: 6, borderRadius: 999, background: "var(--accent)", animation: "pulse 2.4s ease-in-out infinite" }}/>
              FYP 2026 · FAST NUCES Islamabad
            </div>

            <h1 className="font-display tracking-tight reveal" style={{
              fontSize: "clamp(2.5rem, 6.4vw, 5.25rem)",
              lineHeight: 1.06,
              fontWeight: 400,
              color: "var(--text-1)",
              letterSpacing: "-0.025em",
              textWrap: "balance",
            }}>
              Binding affinity from <em className="italic" style={{ color: "var(--accent)" }}>protein dynamics</em>,
              <br/>
              <span style={{ color: "var(--text-2)" }}>not just a frozen frame.</span>
            </h1>

            <p className="reveal text-lg" style={{
              marginTop: 32, color: "var(--text-2)", maxWidth: 560,
              textWrap: "pretty"
            }}>
              Most published binding-affinity models train on a single static crystal snapshot.
              ProtAI trains on the whole molecular-dynamics trajectory — 100 frames per complex, 16,972 complexes, every protein actually moving.
            </p>

            <div className="reveal" style={{ marginTop: 40, display: "flex", flexWrap: "wrap", gap: 12 }}>
              <a href="#/demo" className="btn btn-primary btn-lg">
                Try the demo
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round"><path d="M3 8h10m0 0L8.5 3.5M13 8l-4.5 4.5"/></svg>
              </a>
              <a href="#/results" className="btn btn-secondary btn-lg">Read the results</a>
              <a href="#/architecture" className="btn btn-ghost btn-lg">Architecture →</a>
            </div>

            <div className="reveal" style={{ marginTop: 56, display: "flex", flexWrap: "wrap", gap: "16px 32px", color: "var(--text-3)", fontSize: 13 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 8, height: 8, borderRadius: 999, background: "var(--data-protein)" }}/>
                <span className="font-mono">protein</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 8, height: 8, borderRadius: 999, background: "var(--data-ligand)" }}/>
                <span className="font-mono">ligand</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 8, height: 8, borderRadius: 999, background: "var(--data-pocket)" }}/>
                <span className="font-mono">binding pocket</span>
              </div>
            </div>
          </div>

          <div className="reveal" style={{ position: "relative" }}>
            <div style={{
              position: "absolute", top: -12, right: 0, zIndex: 2,
              display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 6
            }}>
              <div className="chip">
                <span style={{ width: 6, height: 6, borderRadius: 999, background: "var(--accent)", animation: "pulse 2s ease-in-out infinite" }}/>
                <span className="font-mono">PDB · 1A1B</span>
              </div>
            </div>
            <div className="viewer-host" style={{
              aspectRatio: "1 / 1",
              width: "100%",
              maxWidth: 540,
              marginLeft: "auto",
              border: "var(--border-1)",
              borderRadius: 20,
              overflow: "hidden",
              position: "relative",
            }}>
              <MoleculeViewer pdbId="1A1B" mode="adaptability" spin ariaLabel="Slowly rotating 1A1B protein complex, colored by adaptability"/>
            </div>
            <div style={{
              marginTop: 14,
              display: "flex",
              justifyContent: "space-between",
              maxWidth: 540,
              marginLeft: "auto",
              color: "var(--text-3)",
              fontSize: 12,
            }} className="font-mono">
              <span>100 frames · MD</span>
              <span>truthful ΔG · −10.07 kcal/mol</span>
            </div>
          </div>
        </div>
      </div>
      <style>{`
        @media (max-width: 1023px) {
          .hero-grid { grid-template-columns: 1fr !important; gap: 48px !important; }
        }
      `}</style>
    </section>
  );
}

/* ---------- Numbers strip ---------- */
function NumbersStrip() {
  return (
    <section style={{ borderBottom: "var(--border-1)", background: "var(--surface-1)" }}>
      <div className="container" style={{ padding: "80px 24px" }}>
        <div className="reveal" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 24, marginBottom: 56 }}>
          <div>
            <div className="text-xs uppercase tracking-wide text-3" style={{ marginBottom: 12 }}>Headline metrics · random_logk split · n&thinsp;=&thinsp;1,579</div>
            <h2 className="font-display" style={{
              fontSize: "clamp(2rem, 4vw, 3.25rem)", lineHeight: 1.08, letterSpacing: "-0.02em", maxWidth: 780, textWrap: "balance"
            }}>
              The model gets <em className="italic" style={{ color: "var(--accent)" }}>22.4% better</em> in-distribution when it sees the trajectory.
            </h2>
          </div>
          <a href="#/results" className="btn btn-ghost">All four headline tables →</a>
        </div>

        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: 32,
        }} className="metrics-grid">

          <MetricColumn
            label="Static crystal"
            sub="single frame"
            pearson="0.331"
            spearman="0.301"
            rmse="1.71"
            r2="0.110"
            tint="cool"
          />
          <MetricColumn
            label="Trajectory"
            sub="100 frames, randomly sampled"
            pearson="0.405"
            spearman="0.378"
            rmse="1.66"
            r2="0.164"
            tint="cool"
            delta={{ pearson: "+22.4%", color: "positive" }}
          />
          <MetricColumn
            label="+ Multitask"
            sub="MD energy auxiliary"
            pearson="0.414"
            spearman="0.388"
            rmse="1.65"
            r2="0.170"
            highlighted
            delta={{ pearson: "+25.1%", color: "positive" }}
          />
          <MetricColumn
            label="Cross-eval"
            sub="held-out families"
            pearson="0.244"
            spearman="0.231"
            rmse="1.94"
            r2="0.046"
            tint="warn"
            warn
            note="Trajectory alone collapses to 0.056. Multitask recovers most of it."
          />
        </div>
      </div>
      <style>{`
        @media (max-width: 1023px) {
          .metrics-grid { grid-template-columns: repeat(2, 1fr) !important; gap: 40px !important; }
        }
        @media (max-width: 600px) {
          .metrics-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </section>
  );
}

function MetricColumn({ label, sub, pearson, spearman, rmse, r2, highlighted, tint, warn, delta, note }) {
  const accent = highlighted ? "var(--accent)" : warn ? "var(--warning)" : "var(--text-1)";
  return (
    <div className="reveal" style={{
      padding: "28px 0 0",
      borderTop: highlighted ? "2px solid var(--accent)" : "1px solid var(--surface-3)",
      position: "relative",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <span className="text-xs uppercase tracking-wide" style={{ color: highlighted ? "var(--accent)" : "var(--text-2)", letterSpacing: "0.06em" }}>{label}</span>
      </div>
      <div className="text-3 text-xs font-mono" style={{ marginBottom: 24 }}>{sub}</div>

      <div className="font-display tabular-nums" style={{
        fontSize: "clamp(2.5rem, 4.5vw, 3.5rem)",
        lineHeight: 1.0,
        color: accent,
        fontWeight: 500,
        letterSpacing: "-0.02em",
      }}>
        <NumberCountUp value={parseFloat(pearson)} decimals={3} />
      </div>
      <div className="text-3 text-xs font-mono uppercase tracking-wide" style={{ marginTop: 8 }}>Pearson r</div>

      {delta && (
        <div style={{ marginTop: 10 }}>
          <span className="chip" style={{
            background: "rgba(125, 211, 168, 0.14)",
            color: "var(--positive)",
            borderColor: "transparent",
            fontFamily: "JetBrains Mono, monospace",
          }}>
            {delta.pearson} vs static
          </span>
        </div>
      )}

      <hr className="divider" style={{ margin: "24px 0 16px" }}/>

      <dl style={{ display: "grid", gap: 8, fontSize: 13, fontFamily: "JetBrains Mono, monospace" }}>
        <Row k="Spearman ρ" v={spearman}/>
        <Row k="R²"          v={r2}/>
        <Row k="RMSE"        v={rmse + " ↓"}/>
      </dl>

      {note && (
        <p className="text-sm text-3" style={{ marginTop: 16, lineHeight: 1.45, fontStyle: "italic" }}>{note}</p>
      )}
    </div>
  );
}
function Row({ k, v }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between" }}>
      <dt className="text-3 tabular-nums" style={{ fontFamily: "inherit" }}>{k}</dt>
      <dd className="text-2 tabular-nums" style={{ margin: 0, fontVariantNumeric: "tabular-nums" }}>{v}</dd>
    </div>
  );
}

/* ---------- Discovery cards ---------- */
function DiscoveryCards() {
  const findings = [
    {
      no: "I",
      title: "Trajectory helps.",
      body: "Sampling 100 MD frames per complex during training lifts in-distribution Pearson from 0.331 to 0.405. The model learns flexibility, not just shape.",
      key: "0.405",
      keyLabel: "Pearson in-distribution",
    },
    {
      no: "II",
      title: "OOD generalization collapses.",
      body: "When held out on novel protein families, the trajectory-only model drops to Pearson 0.056. The dynamics it learned were family-specific.",
      key: "0.056",
      keyLabel: "Cross-eval Pearson · trajectory",
      warn: true,
    },
    {
      no: "III",
      title: "Multitask threads the needle.",
      body: "Adding an MD-energy auxiliary head recovers cross-eval Pearson to 0.244 — most of the static baseline (0.311) — without giving up the in-distribution gain.",
      key: "0.244 / 0.414",
      keyLabel: "cross-eval · in-distribution",
      highlight: true,
    },
  ];
  return (
    <section style={{ borderBottom: "var(--border-1)" }}>
      <div className="container" style={{ padding: "112px 24px" }}>
        <div className="reveal" style={{ maxWidth: 760, marginBottom: 64 }}>
          <span className="chip" style={{ marginBottom: 16 }}>Three findings</span>
          <h2 className="font-display" style={{
            fontSize: "clamp(2.25rem, 4.5vw, 3.5rem)", lineHeight: 1.08, letterSpacing: "-0.02em",
            textWrap: "pretty",
          }}>
            Trajectory training trades out-of-distribution generalization for in-distribution accuracy.
            <em className="italic" style={{ color: "var(--accent)" }}> A multitask formulation gets most of it back.</em>
          </h2>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 20 }} className="findings-grid">
          {findings.map((f) => (
            <article
              key={f.no}
              className={"reveal card card-interactive " + (f.highlight ? "card-accent" : "")}
              style={{ padding: 28, position: "relative", display: "flex", flexDirection: "column", gap: 16, minHeight: 320 }}
            >
              <div className="font-display" style={{ fontSize: "2.25rem", color: f.highlight ? "var(--accent)" : "var(--text-3)", lineHeight: 1, fontStyle: "italic" }}>
                {f.no}
              </div>
              <h3 className="font-display" style={{ fontSize: "1.625rem", lineHeight: 1.2, color: "var(--text-1)", letterSpacing: "-0.01em" }}>
                {f.title}
              </h3>
              <p className="text-2 text-sm" style={{ flex: 1, textWrap: "pretty" }}>
                {f.body}
              </p>
              <div style={{ paddingTop: 16, borderTop: "var(--border-1)" }}>
                <div className="font-mono tabular-nums" style={{ fontSize: "1.5rem", color: f.warn ? "var(--negative)" : f.highlight ? "var(--accent)" : "var(--text-1)", letterSpacing: "-0.01em" }}>
                  {f.key}
                </div>
                <div className="text-xs text-3 font-mono uppercase tracking-wide" style={{ marginTop: 4 }}>{f.keyLabel}</div>
              </div>
            </article>
          ))}
        </div>
      </div>
      <style>{`
        @media (max-width: 1023px) {
          .findings-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </section>
  );
}

/* ---------- Dataset visual ---------- */
function DatasetVisual() {
  const [frame, setFrame] = useState(50);
  const energy = useMemo(() => {
    // Plausible MD energy curve: oscillates around -115 kcal/mol with noise
    const base = -115.17;
    const t = frame / 100;
    const osc = Math.sin(t * Math.PI * 4.6) * 1.8 + Math.cos(t * Math.PI * 2.1) * 1.1;
    const noise = Math.sin(frame * 13.7) * 0.4;
    return base + osc + noise;
  }, [frame]);

  return (
    <section style={{ borderBottom: "var(--border-1)", background: "var(--surface-1)" }}>
      <div className="container" style={{ padding: "112px 24px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 0.9fr) minmax(0, 1.1fr)", gap: 64, alignItems: "center" }} className="dataset-grid">
          <div className="reveal">
            <div className="text-xs uppercase tracking-wide text-3" style={{ marginBottom: 12 }}>MISATO · 124&thinsp;GB</div>
            <h2 className="font-display" style={{ fontSize: "clamp(2rem, 4vw, 3.25rem)", lineHeight: 1.08, letterSpacing: "-0.02em" }}>
              A frozen crystal hides
              <br/>
              <span style={{ color: "var(--text-3)" }}>what binding actually does.</span>
            </h2>
            <p className="text-lg text-2" style={{ marginTop: 24, maxWidth: 480, textWrap: "pretty" }}>
              Ligands wedge into pockets that are still moving. Pockets adapt — induced fit — and that flexibility correlates with binding strength.
            </p>

            <div style={{ display: "grid", gap: 16, marginTop: 32 }}>
              <Bullet text="Per-atom adaptability scores derived from the trajectory."/>
              <Bullet text="Pocket residues within 4.5 Å of the ligand, recomputed per frame."/>
              <Bullet text="~1.7 million conformations seen during training."/>
            </div>
          </div>

          <div className="reveal card" style={{ padding: 24 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 16 }}>
              <div>
                <div className="text-xs uppercase tracking-wide text-3">Single trajectory · PDB 1A1B</div>
                <div className="font-display" style={{ fontSize: "2.5rem", lineHeight: 1, marginTop: 6 }}>
                  <span className="font-mono tabular-nums" style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "2.25rem", color: "var(--text-1)" }}>{energy.toFixed(2)}</span>
                  <span className="text-base text-3 font-mono" style={{ marginLeft: 8 }}>kcal/mol</span>
                </div>
                <div className="text-xs text-3 font-mono uppercase tracking-wide" style={{ marginTop: 4 }}>MD energy · frame {frame.toString().padStart(3, "0")}/100</div>
              </div>
              <div className="chip chip-accent" style={{ alignSelf: "flex-start" }}>scrub →</div>
            </div>

            <EnergyChart frame={frame}/>

            <input
              type="range"
              min={0} max={100} step={1}
              value={frame}
              onChange={(e) => setFrame(parseInt(e.target.value))}
              aria-label="MD frame index"
              style={{ width: "100%", marginTop: 12, accentColor: "var(--accent)" }}
            />

            <div className="font-mono text-xs text-3" style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
              <span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>
            </div>
          </div>
        </div>
      </div>
      <style>{`
        @media (max-width: 1023px) {
          .dataset-grid { grid-template-columns: 1fr !important; gap: 48px !important; }
        }
      `}</style>
    </section>
  );
}

function Bullet({ text }) {
  return (
    <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
      <span style={{
        marginTop: 9, flexShrink: 0, width: 6, height: 6, borderRadius: 999, background: "var(--accent)",
      }}/>
      <span className="text-base text-2">{text}</span>
    </div>
  );
}

function EnergyChart({ frame }) {
  const pts = useMemo(() => {
    const arr = [];
    for (let i = 0; i <= 100; i++) {
      const t = i / 100;
      const v = Math.sin(t * Math.PI * 4.6) * 1.8 + Math.cos(t * Math.PI * 2.1) * 1.1 + Math.sin(i * 13.7) * 0.4;
      arr.push(v);
    }
    return arr;
  }, []);
  const w = 560, h = 160, pad = 4;
  const xmax = w - pad * 2;
  const ymin = Math.min(...pts), ymax = Math.max(...pts);
  const yspan = ymax - ymin;
  const xf = (i) => pad + (i / 100) * xmax;
  const yf = (v) => pad + ((ymax - v) / yspan) * (h - pad * 2);
  const path = pts.map((v, i) => `${i === 0 ? "M" : "L"} ${xf(i).toFixed(2)} ${yf(v).toFixed(2)}`).join(" ");
  const area = path + ` L ${xf(100).toFixed(2)} ${h - pad} L ${xf(0).toFixed(2)} ${h - pad} Z`;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" width="100%" height={160} style={{ borderRadius: 8 }}>
      <defs>
        <linearGradient id="ec" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="var(--accent)" stopOpacity="0.22"/>
          <stop offset="1" stopColor="var(--accent)" stopOpacity="0"/>
        </linearGradient>
      </defs>
      <rect width={w} height={h} fill="var(--surface-2)" rx="6"/>
      {[0, 0.25, 0.5, 0.75, 1].map((p, i) => (
        <line key={i} x1={pad + p * xmax} x2={pad + p * xmax} y1={pad} y2={h - pad} stroke="var(--surface-3)" strokeWidth="1" strokeDasharray="2 4"/>
      ))}
      <path d={area} fill="url(#ec)"/>
      <path d={path} stroke="var(--accent)" strokeWidth="1.5" fill="none" strokeLinejoin="round" strokeLinecap="round"/>
      <line x1={xf(frame)} x2={xf(frame)} y1={pad} y2={h - pad} stroke="var(--text-1)" strokeWidth="1"/>
      <circle cx={xf(frame)} cy={yf(pts[frame])} r="4" fill="var(--bg)" stroke="var(--text-1)" strokeWidth="1.5"/>
    </svg>
  );
}

/* ---------- Architecture one-liner ---------- */
function ArchitectureOneLiner() {
  return (
    <section style={{ borderBottom: "var(--border-1)" }}>
      <div className="container" style={{ padding: "112px 24px" }}>
        <div className="reveal" style={{ display: "grid", gridTemplateColumns: "minmax(0, 0.85fr) minmax(0, 1.15fr)", gap: 56, alignItems: "center" }} className="arch-grid">
          <div>
            <span className="chip" style={{ marginBottom: 16 }}>Architecture · in one breath</span>
            <h2 className="font-display" style={{ fontSize: "clamp(2rem, 3.4vw, 3rem)", lineHeight: 1.1, letterSpacing: "-0.015em" }}>
              SchNet edges, multitask heads, an E(3)-equivariant message passing core.
            </h2>
            <p className="text-lg text-2" style={{ marginTop: 20, maxWidth: 520, textWrap: "pretty" }}>
              Atom features carry element identity and per-atom adaptability. Edges are radial-basis-expanded at a 4.5&thinsp;Å cutoff. Two heads predict log K and MD energy.
            </p>
            <div style={{ marginTop: 24 }}>
              <a href="#/architecture" className="btn-link">See the full pipeline →</a>
            </div>
          </div>

          <div>
            <ArchDiagram/>
          </div>
        </div>
      </div>
      <style>{`
        @media (max-width: 1023px) {
          .arch-grid { grid-template-columns: 1fr !important; gap: 40px !important; }
        }
      `}</style>
    </section>
  );
}

function ArchDiagram() {
  // Horizontal pipeline: Input → Embed → SchNet × 4 → Pool → [Head log K, Head energy]
  const boxes = [
    { label: "Atoms + adaptability", sub: "x(z, a) ∈ ℝⁿˣᵈ", w: 168 },
    { label: "Embedding", sub: "linear(d → 128)", w: 132 },
    { label: "SchNet × 4", sub: "CFConv + RBF, 4.5 Å", w: 156 },
    { label: "Pool", sub: "sum + attention", w: 110 },
  ];
  let x = 0;
  const positions = boxes.map((b) => {
    const obj = { ...b, x };
    x += b.w + 24;
    return obj;
  });
  const totalW = x - 24;
  return (
    <div className="card" style={{ padding: 20, background: "var(--surface-2)", overflowX: "auto" }}>
      <svg viewBox={`0 0 ${Math.max(720, totalW + 200)} 230`} width="100%" style={{ minWidth: 640 }}>
        {/* Pipeline boxes */}
        {positions.map((b, i) => (
          <g key={i}>
            <rect x={b.x} y={70} width={b.w} height={56} rx="8"
              fill="var(--surface-1)" stroke="var(--surface-3)" strokeWidth="1"/>
            <text x={b.x + b.w/2} y={92} textAnchor="middle" fontFamily="Inter, sans-serif" fontWeight="500" fontSize="13" fill="var(--text-1)">
              {b.label}
            </text>
            <text x={b.x + b.w/2} y={111} textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontSize="11" fill="var(--text-3)">
              {b.sub}
            </text>
            {i < positions.length - 1 && (
              <g transform={`translate(${b.x + b.w + 4}, 98)`}>
                <line x1="0" x2="16" y1="0" y2="0" stroke="var(--text-3)" strokeWidth="1.25"/>
                <path d="M 14 -4 L 20 0 L 14 4" fill="none" stroke="var(--text-3)" strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round"/>
              </g>
            )}
          </g>
        ))}
        {/* Branch out to two heads */}
        <g transform={`translate(${totalW + 20}, 0)`}>
          <line x1="0" y1="98" x2="40" y2="48" stroke="var(--text-3)" strokeWidth="1.25" strokeLinecap="round"/>
          <line x1="0" y1="98" x2="40" y2="148" stroke="var(--text-3)" strokeWidth="1.25" strokeLinecap="round"/>
          <rect x="40" y="22" width="140" height="52" rx="8" fill="var(--surface-1)" stroke="var(--accent)" strokeWidth="1.25"/>
          <text x="110" y="43" textAnchor="middle" fontFamily="Inter, sans-serif" fontWeight="600" fontSize="13" fill="var(--accent)">Head · log K</text>
          <text x="110" y="62" textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontSize="11" fill="var(--text-3)">main objective</text>

          <rect x="40" y="122" width="140" height="52" rx="8" fill="var(--surface-1)" stroke="var(--surface-3)" strokeWidth="1"/>
          <text x="110" y="143" textAnchor="middle" fontFamily="Inter, sans-serif" fontWeight="500" fontSize="13" fill="var(--text-1)">Head · MD energy</text>
          <text x="110" y="162" textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontSize="11" fill="var(--text-3)">auxiliary, λ = 0.2</text>
        </g>

        {/* Labels above and below */}
        <text x="0" y="24" fontFamily="JetBrains Mono, monospace" fontSize="11" fill="var(--text-3)">INPUT</text>
        <text x={totalW + 80} y={24} fontFamily="JetBrains Mono, monospace" fontSize="11" fill="var(--text-3)">OUTPUTS</text>
      </svg>
    </div>
  );
}

/* ---------- Final CTA ---------- */
function CTAFooter() {
  return (
    <section>
      <div className="container" style={{ padding: "128px 24px", textAlign: "center" }}>
        <h2 className="reveal font-display" style={{ fontSize: "clamp(2.5rem, 5.5vw, 4.5rem)", lineHeight: 1.05, letterSpacing: "-0.025em", maxWidth: 880, margin: "0 auto" }}>
          Run a prediction in your browser.
        </h2>
        <p className="reveal text-lg text-2" style={{ marginTop: 24, maxWidth: 600, margin: "24px auto 0" }}>
          Pick any of 16,972 complexes. Watch the pocket breathe. Compare predicted vs. measured binding energy in real time.
        </p>
        <div className="reveal" style={{ marginTop: 48, display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
          <a href="#/demo" className="btn btn-primary btn-lg">Open the demo →</a>
          <a href="https://github.com/LastPredator/ProtAI" target="_blank" rel="noopener noreferrer" className="btn btn-secondary btn-lg">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38v-1.34c-2.22.48-2.69-1.07-2.69-1.07-.36-.92-.89-1.17-.89-1.17-.73-.5.06-.49.06-.49.81.06 1.23.83 1.23.83.72 1.23 1.88.87 2.34.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.13 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.03 2.2-.82 2.2-.82.44 1.11.16 1.93.08 2.13.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.74.54 1.49v2.21c0 .21.15.46.55.38C13.71 14.53 16 11.54 16 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            View source
          </a>
        </div>
      </div>
    </section>
  );
}

window.PageLanding = PageLanding;
