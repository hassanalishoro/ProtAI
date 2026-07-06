/* ============================================================
   Content pages — Architecture, Methodology, Results, Reference
   ============================================================ */

/* ===========================================================
   ARCHITECTURE
   =========================================================== */
function PageArchitecture() {
  useReveal();
  return (
    <article>
      <ContentHeader
        chip="Architecture"
        title={<>The pipeline, in one breath: <em className="italic" style={{ color: "var(--accent)" }}>SchNet edges, multitask heads, equivariant core.</em></>}
        sub="From PDB ID to predicted log K — how data flows, where each loss lives, and the few hyperparameters that mattered."
      />

      <Section paddingY="md">
        <SubsectionTitle eyebrow="Step 01" title="From PDB to graph"/>
        <Prose>
          Each protein–ligand complex enters the pipeline as a list of atom records sourced from MISATO. We strip waters, keep heavy atoms only, and merge the protein and ligand into a single graph. Nodes carry two features: the atomic number embedded in 16 dimensions, and a per-atom adaptability score computed from the molecular-dynamics trajectory.
        </Prose>
        <CodeBlock lines={[
          ['# atom features per node', 'c-comment'],
          ['x = concat[ embed(atomic_number, 16),  adaptability_scalar ]', ''],
          ['x.shape  # → (n_atoms, 17)', 'c-comment'],
        ]}/>
        <Prose>
          Edges connect any two atoms within 4.5&thinsp;Å, recomputed once per frame. Edge attributes are the raw Euclidean distance expanded by a Gaussian radial basis — the same trick used by the original SchNet paper to make the distance signal differentiable and resolution-flexible.
        </Prose>
      </Section>

      <Section paddingY="md" reveal={false}>
        <div className="reveal">
          <SubsectionTitle eyebrow="Step 02" title="Equivariant message passing"/>
        </div>
        <PipelineDiagramFull/>
        <Prose>
          Four SchNet continuous-filter convolution blocks (CFConv) stack on top of the embedding. Each block computes per-edge messages from the radial-basis distance, sums them onto each node, and adds the residual. The design is <strong>E(3)-equivariant</strong> by construction — rotations and translations of the input produce the same outputs. We don&apos;t learn invariance, we get it.
        </Prose>
        <HyperparamTable/>
      </Section>

      <Section paddingY="md">
        <SubsectionTitle eyebrow="Step 03" title="Multitask heads, one body"/>
        <Prose>
          A pooled graph embedding feeds two linear heads. The first regresses the experimental binding affinity (log K from PDBbind+). The second regresses the molecular-dynamics energy at the same frame. The two heads share parameters everywhere except the final 256→1 projection.
        </Prose>
        <LossEquation/>
        <Prose>
          The MD-energy auxiliary head is the smaller of the two losses (weighted at λ&thinsp;=&thinsp;0.2) but does most of the regularizing. Without it, the trajectory-trained model overfits to family-specific dynamics; with it, cross-evaluation Pearson recovers from 0.056 to 0.244.
        </Prose>
        <Callout
          title="Why it works (we think)"
          body="The MD energy is a physical signal independent of the binding label. By forcing the network to also predict it, we anchor the learned representation to physics rather than to family-specific structural cues."
        />
      </Section>

      <Section paddingY="md">
        <SubsectionTitle eyebrow="Step 04" title="What we deferred"/>
        <Prose>
          Three things we considered and did not ship in this iteration. Each is documented in the GitHub README under <span className="font-mono">FUTURE.md</span>.
        </Prose>
        <DeferredList/>

        <div style={{ marginTop: 56, padding: 24, background: "var(--surface-1)", borderRadius: 14, border: "var(--border-1)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 16 }}>
            <div>
              <div className="text-xs uppercase tracking-wide text-3" style={{ marginBottom: 6 }}>Next</div>
              <h3 className="font-display" style={{ fontSize: "1.5rem" }}>See it run on a real complex.</h3>
            </div>
            <a href="#/demo" className="btn btn-primary">Open the demo →</a>
          </div>
        </div>
      </Section>
    </article>
  );
}

function PipelineDiagramFull() {
  return <AnimatedPipeline/>;
}

/* ---------- Animated forward pass through the network ---------- */
const PIPE_STEPS = [
  {
    id: "input",
    label: "Input",
    sub: "atoms + adaptability",
    shape: "(n, 17)",
    body: "14 heavy atoms enter the graph. Each carries an atomic number and a scalar adaptability score derived from the trajectory.",
    sample: "x[0] = [Z=6, a=0.42, …]\nx[1] = [Z=7, a=0.18, …]",
  },
  {
    id: "edges",
    label: "Edges",
    sub: "radial cutoff 4.5 Å",
    shape: "(e, 2)",
    body: "Any two atoms within 4.5 Å form an edge. Edge attributes are the raw Euclidean distance expanded into 128 Gaussian bases.",
    sample: "edges = 43\nd[0] = 1.51 Å → RBF₁₂₈",
  },
  {
    id: "embed",
    label: "Embedding",
    sub: "17 → 128",
    shape: "(n, 128)",
    body: "A single linear layer lifts each 17-d node feature into a 128-d hidden state. After this the network is fully differentiable in the hidden space.",
    sample: "h[0] ∈ ℝ¹²⁸\n‖h[0]‖ ≈ 0.92",
  },
  {
    id: "schnet1",
    label: "SchNet 1/4",
    sub: "CFConv + RBF",
    shape: "(n, 128)",
    body: "Each node aggregates messages from its 4.5 Å neighbors. Messages are a learned function of distance only — E(3) equivariance by construction.",
    sample: "msg = φ(RBF(d)) ⊙ W·h_j\nh ← h + Σ msg",
  },
  {
    id: "schnet2",
    label: "SchNet 2/4",
    sub: "deeper context",
    shape: "(n, 128)",
    body: "Second hop. Information from 2-bond neighbors now reaches each node. Pocket residues start to feel ligand atoms across the cleft.",
    sample: "receptive field ≈ 9 Å",
  },
  {
    id: "schnet3",
    label: "SchNet 3/4",
    sub: "long-range mix",
    shape: "(n, 128)",
    body: "Third block. The graph radius is ~13.5 Å — wide enough that flexibility correlations across the pocket are visible to every node.",
    sample: "receptive field ≈ 13.5 Å",
  },
  {
    id: "schnet4",
    label: "SchNet 4/4",
    sub: "final mix",
    shape: "(n, 128)",
    body: "Last block. Node hidden states are now context-aware vectors that encode local chemistry + global pocket geometry.",
    sample: "h_final = SchNet(h_3, edges)",
  },
  {
    id: "pool",
    label: "Pool",
    sub: "Σ + attention",
    shape: "(128)",
    body: "Sum-aggregation followed by a small attention head. The graph collapses into a single 128-d vector summarizing the whole complex.",
    sample: "g = Σᵢ αᵢ · hᵢ",
  },
  {
    id: "heads",
    label: "Heads",
    sub: "log K · MD energy",
    shape: "(1) · (1)",
    body: "Two linear projections, fully separate. The primary head regresses log K (PDBbind label). The auxiliary head regresses MD energy. λ = 0.2.",
    sample: "log K̂ = −10.07\nÊ_MD = −115.17 kcal/mol",
  },
];

function AnimatedPipeline() {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1); // 0.5, 1, 2

  // Auto-advance
  useEffect(() => {
    if (!playing) return;
    const interval = 1700 / speed;
    const id = setInterval(() => {
      setStep((s) => (s + 1) % PIPE_STEPS.length);
    }, interval);
    return () => clearInterval(id);
  }, [playing, speed]);

  const stage = PIPE_STEPS[step];

  return (
    <div className="card" style={{ padding: 0, marginBottom: 32, overflow: "hidden", background: "var(--surface-1)" }}>
      {/* Pipeline header strip */}
      <PipelineHeader step={step} onStep={setStep}/>

      {/* Main viz area: graph on left, info on right */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "minmax(0, 1.35fr) minmax(0, 1fr)",
        gap: 0,
        borderTop: "var(--border-1)",
        borderBottom: "var(--border-1)",
      }} className="anim-pipe-grid">
        <div style={{ borderRight: "var(--border-1)", background: "var(--surface-2)", position: "relative", minHeight: 360 }} className="anim-pipe-canvas">
          <GraphCanvas step={step}/>
          <StageBadge stage={stage}/>
          <ParticleField active={playing} step={step}/>
        </div>
        <StageInfoPanel stage={stage} stepIdx={step}/>
      </div>

      {/* Controls */}
      <PipelineControls
        playing={playing} onPlay={() => setPlaying((p) => !p)}
        speed={speed} onSpeed={setSpeed}
        step={step} total={PIPE_STEPS.length}
        onStep={setStep}
      />

      <style>{`
        @media (max-width: 880px) {
          .anim-pipe-grid { grid-template-columns: 1fr !important; }
          .anim-pipe-canvas { border-right: 0 !important; border-bottom: var(--border-1) !important; }
        }
      `}</style>
    </div>
  );
}

/* Top header — 9 segments, click to jump */
function PipelineHeader({ step, onStep }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(${PIPE_STEPS.length}, 1fr)`, gap: 0 }}>
      {PIPE_STEPS.map((s, i) => {
        const active = i === step;
        const done = i < step;
        return (
          <button
            key={s.id}
            onClick={() => onStep(i)}
            style={{
              padding: "14px 10px",
              borderRight: i < PIPE_STEPS.length - 1 ? "var(--border-1)" : "none",
              background: active ? "var(--accent-soft)" : "transparent",
              borderBottom: active ? "2px solid var(--accent)" : "2px solid transparent",
              color: active ? "var(--accent)" : done ? "var(--text-2)" : "var(--text-3)",
              textAlign: "left",
              cursor: "pointer",
              transition: "background 200ms, color 200ms, border-color 200ms",
            }}
            aria-current={active ? "step" : undefined}
          >
            <div className="font-mono text-xs uppercase tracking-wide" style={{ marginBottom: 4, opacity: 0.85 }}>
              {String(i + 1).padStart(2, "0")}
            </div>
            <div style={{ fontSize: 13, fontWeight: 500, lineHeight: 1.1 }}>{s.label}</div>
          </button>
        );
      })}
    </div>
  );
}

/* Fixed-layout molecule graph with edges + animated message passing */
const GRAPH_ATOMS = [
  // Protein backbone (loose)
  { i: 0,  x: 80,  y: 80,  z: 6, kind: "C", role: "prot" },
  { i: 1,  x: 130, y: 60,  z: 7, kind: "N", role: "prot" },
  { i: 2,  x: 175, y: 95,  z: 6, kind: "C", role: "prot" },
  { i: 3,  x: 165, y: 155, z: 8, kind: "O", role: "prot" },
  { i: 4,  x: 90,  y: 170, z: 6, kind: "C", role: "prot" },
  { i: 5,  x: 130, y: 215, z: 7, kind: "N", role: "prot" },
  // Pocket residues
  { i: 6,  x: 230, y: 140, z: 6, kind: "C", role: "pock" },
  { i: 7,  x: 265, y: 100, z: 16, kind: "S", role: "pock" },
  { i: 8,  x: 285, y: 165, z: 7, kind: "N", role: "pock" },
  // Ligand
  { i: 9,  x: 360, y: 140, z: 6, kind: "C", role: "lig" },
  { i: 10, x: 410, y: 110, z: 8, kind: "O", role: "lig" },
  { i: 11, x: 425, y: 175, z: 6, kind: "C", role: "lig" },
  { i: 12, x: 470, y: 145, z: 7, kind: "N", role: "lig" },
  { i: 13, x: 390, y: 200, z: 6, kind: "C", role: "lig" },
];

const GRAPH_EDGES = [
  [0,1],[1,2],[2,3],[0,4],[4,5],
  [2,6],[6,7],[6,8],[3,6],
  [8,9],[9,10],[9,11],[10,12],[11,13],[11,12],
  // cross-pocket "binding" edges
  [6,9],[8,9],[7,9],
];

const ATOM_COLOR = { C: "#94A3B8", N: "#60A5FA", O: "#F87171", S: "#FCD34D", P: "#A78BFA" };
const ROLE_COLOR = { prot: "var(--data-protein)", pock: "var(--data-pocket)", lig: "var(--data-ligand)" };

function GraphCanvas({ step }) {
  const stageId = PIPE_STEPS[step].id;
  const inSchNet = stageId.startsWith("schnet");
  const schnetIdx = inSchNet ? parseInt(stageId.replace("schnet", ""), 10) - 1 : -1;

  // Receptive-field radius grows with each SchNet block
  const radius = inSchNet ? 65 + schnetIdx * 35 : 0;

  // Sample atom for receptive field visualization (atom 9 — central ligand carbon)
  const focusAtom = GRAPH_ATOMS[9];

  // Which atoms get "lit" depends on stage
  const litAtoms = useMemo(() => {
    if (stageId === "input")  return new Set(GRAPH_ATOMS.map(a => a.i));
    if (stageId === "edges")  return new Set(GRAPH_ATOMS.map(a => a.i));
    if (stageId === "embed")  return new Set(GRAPH_ATOMS.map(a => a.i));
    if (inSchNet) {
      // Atoms within receptive field of the focus atom
      const set = new Set();
      GRAPH_ATOMS.forEach((a) => {
        const d = Math.hypot(a.x - focusAtom.x, a.y - focusAtom.y);
        if (d <= radius) set.add(a.i);
      });
      return set;
    }
    if (stageId === "pool")  return new Set(GRAPH_ATOMS.map(a => a.i));
    if (stageId === "heads") return new Set();
    return new Set();
  }, [stageId, radius, focusAtom.x, focusAtom.y]);

  const showColorsByElement = stageId === "input";
  const showColorsByRole    = ["edges", "embed", "pool", "heads"].includes(stageId);
  const showHidden          = inSchNet;

  // Pool: animate atoms collapsing into a single center vector
  const poolProgress = stageId === "pool" ? 1 : 0;
  // Heads: show two numeric output projections coming out
  const heads = stageId === "heads";

  return (
    <svg viewBox="0 0 540 320" width="100%" style={{ display: "block" }}>
      <defs>
        <radialGradient id="rfg">
          <stop offset="0" stopColor="var(--accent)" stopOpacity="0.18"/>
          <stop offset="1" stopColor="var(--accent)" stopOpacity="0"/>
        </radialGradient>
        <linearGradient id="msggrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="var(--accent)" stopOpacity="0"/>
          <stop offset="0.5" stopColor="var(--accent)" stopOpacity="1"/>
          <stop offset="1" stopColor="var(--accent)" stopOpacity="0"/>
        </linearGradient>
      </defs>

      {/* Receptive-field circle (SchNet) */}
      {inSchNet && (
        <circle
          cx={focusAtom.x} cy={focusAtom.y} r={radius}
          fill="url(#rfg)"
          stroke="var(--accent)" strokeWidth="0.75" strokeDasharray="3 4" opacity="0.7"
          style={{ transition: "r 600ms cubic-bezier(0.22, 1, 0.36, 1)" }}
        />
      )}

      {/* Edges */}
      {GRAPH_EDGES.map(([a, b], idx) => {
        const A = GRAPH_ATOMS[a], B = GRAPH_ATOMS[b];
        const visible = stageId !== "input";
        const active = inSchNet && litAtoms.has(a) && litAtoms.has(b);
        return (
          <g key={idx}>
            <line
              x1={A.x} y1={A.y} x2={B.x} y2={B.y}
              stroke={active ? "var(--accent)" : "var(--surface-3)"}
              strokeWidth={active ? 1.25 : 0.75}
              opacity={visible ? (active ? 0.95 : 0.4) : 0}
              style={{ transition: "opacity 350ms, stroke 350ms, stroke-width 350ms" }}
            />
            {/* Message-passing pulses */}
            {active && (
              <MessagePulse a={A} b={B} schnetIdx={schnetIdx}/>
            )}
          </g>
        );
      })}

      {/* Pool collapsing target */}
      {(stageId === "pool" || heads) && (
        <circle cx={270} cy={160} r="12"
          fill="var(--accent)" opacity="0.25"
          style={{ animation: "pulse 2s ease-in-out infinite" }}/>
      )}
      {(stageId === "pool" || heads) && (
        <circle cx={270} cy={160} r="6" fill="var(--accent)"/>
      )}

      {/* Atoms */}
      {GRAPH_ATOMS.map((a) => {
        const lit = litAtoms.has(a.i);
        let fill;
        if (showColorsByElement) fill = ATOM_COLOR[a.kind] || "var(--text-3)";
        else if (showColorsByRole) fill = ROLE_COLOR[a.role];
        else fill = "var(--accent)";

        // Pool collapse: atoms travel toward center (270, 160)
        const tx = stageId === "pool" ? a.x + (270 - a.x) * 0.55 : a.x;
        const ty = stageId === "pool" ? a.y + (160 - a.y) * 0.55 : a.y;
        const opacity = heads ? 0.12 : (showHidden && !lit ? 0.25 : 0.95);
        const r = stageId === "input" ? 6.5 : showHidden ? (lit ? 7 : 5) : 6.5;

        return (
          <g key={a.i} style={{ transition: "transform 600ms cubic-bezier(0.22, 1, 0.36, 1), opacity 350ms" }}>
            <circle
              cx={tx} cy={ty} r={r}
              fill={fill} opacity={opacity}
              stroke={a.i === focusAtom.i && inSchNet ? "var(--accent)" : "none"}
              strokeWidth={a.i === focusAtom.i && inSchNet ? 1.5 : 0}
              style={{ transition: "cx 600ms cubic-bezier(0.22, 1, 0.36, 1), cy 600ms cubic-bezier(0.22, 1, 0.36, 1), r 350ms" }}
            />
            {showHidden && lit && (
              <circle cx={tx} cy={ty} r={r + 4}
                fill="none" stroke="var(--accent)" strokeWidth="1" opacity="0.35"/>
            )}
          </g>
        );
      })}

      {/* Heads — two output projections */}
      {heads && (
        <g>
          <line x1={282} y1={160} x2={420} y2={100} stroke="var(--accent)" strokeWidth="1.25" strokeDasharray="2 3"/>
          <line x1={282} y1={160} x2={420} y2={220} stroke="var(--text-3)" strokeWidth="1" strokeDasharray="2 3"/>

          <g transform="translate(420, 78)">
            <rect width={104} height={42} rx="6" fill="var(--surface-1)" stroke="var(--accent)" strokeWidth="1.25"/>
            <text x="10" y="17" fontFamily="JetBrains Mono, monospace" fontSize="10" fill="var(--text-3)">log K̂</text>
            <text x="10" y="34" fontFamily="JetBrains Mono, monospace" fontSize="14" fontWeight="600" fill="var(--accent)">−10.07</text>
          </g>
          <g transform="translate(420, 198)">
            <rect width={104} height={42} rx="6" fill="var(--surface-1)" stroke="var(--surface-3)" strokeWidth="1"/>
            <text x="10" y="17" fontFamily="JetBrains Mono, monospace" fontSize="10" fill="var(--text-3)">Ê (kcal/mol)</text>
            <text x="10" y="34" fontFamily="JetBrains Mono, monospace" fontSize="14" fontWeight="500" fill="var(--text-1)">−115.17</text>
          </g>
        </g>
      )}
    </svg>
  );
}

/* A pulse travels A→B along the edge to show message direction */
function MessagePulse({ a, b, schnetIdx }) {
  const dur = 0.9 - schnetIdx * 0.1;
  return (
    <circle r="2.4" fill="var(--accent)">
      <animateMotion dur={`${dur}s`} repeatCount="indefinite" path={`M ${a.x} ${a.y} L ${b.x} ${b.y}`}/>
      <animate attributeName="opacity" values="0;1;0" dur={`${dur}s`} repeatCount="indefinite"/>
    </circle>
  );
}

/* Floating particles that streak across the canvas during transitions */
function ParticleField({ active, step }) {
  // CSS-only; particles tinted by accent
  const particles = [];
  for (let i = 0; i < 6; i++) {
    particles.push(
      <span key={i} style={{
        position: "absolute",
        width: 3, height: 3, borderRadius: 999,
        background: "var(--accent)",
        opacity: 0.55,
        top: `${15 + i * 12}%`, left: -8,
        animation: active ? `streak ${2.2 + i * 0.3}s linear infinite` : "none",
        animationDelay: `${i * 0.4}s`,
        pointerEvents: "none",
      }}/>
    );
  }
  return (
    <div style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "hidden" }}>
      {particles}
      <style>{`
        @keyframes streak {
          0%   { transform: translateX(0)     scale(1);   opacity: 0; }
          15%  { opacity: 0.6; }
          85%  { opacity: 0.6; }
          100% { transform: translateX(560px) scale(0.4); opacity: 0; }
        }
      `}</style>
    </div>
  );
}

/* Bottom-left badge that names the current stage */
function StageBadge({ stage }) {
  return (
    <div style={{
      position: "absolute", left: 16, top: 16,
      display: "flex", flexDirection: "column", gap: 4,
      pointerEvents: "none",
    }}>
      <div className="font-mono text-xs uppercase tracking-wide" style={{ color: "var(--text-3)" }}>
        forward pass · stage
      </div>
      <div className="font-display" style={{ fontSize: "1.5rem", lineHeight: 1, color: "var(--text-1)" }}>
        {stage.label}
      </div>
      <div className="font-mono text-xs" style={{ color: "var(--text-3)" }}>{stage.sub}</div>
    </div>
  );
}

/* Right panel — tensor shape + body + sample */
function StageInfoPanel({ stage, stepIdx }) {
  // Reset the animation each time the stage changes so explanation re-fades
  const [, setKey] = useState(0);
  useEffect(() => { setKey((k) => k + 1); }, [stepIdx]);

  return (
    <div style={{ padding: 28, display: "flex", flexDirection: "column", gap: 22, minHeight: 360 }}>
      <div>
        <div className="text-xs uppercase tracking-wide text-3 font-mono" style={{ marginBottom: 8 }}>Tensor</div>
        <div key={"shape-" + stepIdx} className="font-mono tabular-nums" style={{
          fontSize: "2rem", color: "var(--accent)", letterSpacing: "-0.01em", lineHeight: 1,
          animation: "fadeUp 400ms cubic-bezier(0.16, 1, 0.3, 1) both",
        }}>
          {stage.shape}
        </div>
      </div>

      <div>
        <div className="text-xs uppercase tracking-wide text-3 font-mono" style={{ marginBottom: 8 }}>What happens here</div>
        <p key={"body-" + stepIdx} className="text-base text-2" style={{
          textWrap: "pretty", maxWidth: 460,
          animation: "fadeUp 500ms cubic-bezier(0.16, 1, 0.3, 1) both",
        }}>{stage.body}</p>
      </div>

      <div style={{ marginTop: "auto" }}>
        <div className="text-xs uppercase tracking-wide text-3 font-mono" style={{ marginBottom: 8 }}>Sample</div>
        <pre
          key={"sample-" + stepIdx}
          className="font-mono"
          style={{
            background: "var(--surface-2)",
            border: "var(--border-1)",
            borderRadius: 8,
            padding: "12px 14px",
            fontSize: 12,
            lineHeight: 1.6,
            color: "var(--text-1)",
            whiteSpace: "pre-wrap",
            animation: "fadeUp 600ms cubic-bezier(0.16, 1, 0.3, 1) both",
          }}>{stage.sample}</pre>
      </div>
    </div>
  );
}

/* Bottom — play/pause, scrubber, speed */
function PipelineControls({ playing, onPlay, speed, onSpeed, step, total, onStep }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, padding: "14px 20px", flexWrap: "wrap" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <button
          onClick={onPlay}
          className="btn btn-secondary btn-sm"
          style={{ width: 84, justifyContent: "center" }}
          aria-label={playing ? "Pause animation" : "Play animation"}
        >
          {playing ? (
            <>
              <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor" aria-hidden="true"><rect x="1" y="0" width="3" height="10" rx="0.5"/><rect x="6" y="0" width="3" height="10" rx="0.5"/></svg>
              Pause
            </>
          ) : (
            <>
              <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor" aria-hidden="true"><path d="M1 0v10l9-5z"/></svg>
              Play
            </>
          )}
        </button>
        <button onClick={() => onStep((step - 1 + total) % total)} className="btn btn-ghost btn-sm" aria-label="Previous step">←</button>
        <button onClick={() => onStep((step + 1) % total)}         className="btn btn-ghost btn-sm" aria-label="Next step">→</button>
        <span className="font-mono text-xs text-3 tabular-nums" style={{ marginLeft: 4 }}>
          {String(step + 1).padStart(2, "0")} / {total}
        </span>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span className="text-xs uppercase tracking-wide text-3">Speed</span>
        <div className="segmented" style={{ borderRadius: 8, padding: 3 }}>
          {[0.5, 1, 2].map((s) => (
            <button
              key={s}
              onClick={() => onSpeed(s)}
              className={s === speed ? "active" : ""}
              style={{ padding: "4px 10px", fontSize: 12 }}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function HyperparamTable() {
  const rows = [
    ["Hidden dim",          "128"],
    ["Number of SchNet blocks", "4"],
    ["Edge cutoff",          "4.5 Å"],
    ["RBF basis size",       "128"],
    ["Pooling",              "Σ + multi-head attention"],
    ["Multitask weight λ",   "0.2"],
    ["Optimizer",            "AdamW, β₁ = 0.9, β₂ = 0.999"],
    ["Learning rate",        "3 × 10⁻⁴ · cosine decay"],
    ["Batch size",           "32 complexes"],
    ["Epochs",               "120"],
    ["Frames sampled per epoch", "1 random of 100, per complex"],
  ];
  return (
    <div style={{ overflowX: "auto", marginTop: 24 }}>
      <table className="table-booktabs">
        <thead><tr><th>Hyperparameter</th><th className="num">Value</th></tr></thead>
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <td className="text-2">{k}</td>
              <td className="num text-1">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LossEquation() {
  return (
    <div className="card" style={{ padding: 24, marginTop: 24, background: "var(--surface-2)" }}>
      <div className="text-xs uppercase tracking-wide text-3" style={{ marginBottom: 12 }}>Joint objective</div>
      <div className="font-display tabular-nums" style={{
        fontSize: "1.5rem",
        fontFamily: "Instrument Serif, Georgia, serif",
        fontStyle: "italic",
        letterSpacing: "-0.005em",
        lineHeight: 1.4,
      }}>
        ℒ&nbsp;=&nbsp;<span style={{ color: "var(--accent)" }}>ℒ<sub>log&thinsp;K</sub></span>
        &nbsp;+&nbsp;<span style={{ color: "var(--text-2)" }}>λ&thinsp;·&thinsp;ℒ<sub>E</sub></span>
        ,&nbsp;&nbsp;λ&nbsp;=&nbsp;0.2
      </div>
      <div className="text-sm text-2" style={{ marginTop: 16, textWrap: "pretty" }}>
        Both losses are mean-squared-error in their native units. The energy loss is scaled to roughly match the magnitude of the affinity loss before the λ weighting.
      </div>
    </div>
  );
}

function DeferredList() {
  const items = [
    {
      n: "01", title: "Self-supervised pretraining",
      body: "Pretrain the SchNet body on per-frame energy alone before fine-tuning on affinity. We expect this to lift cross-evaluation Pearson further but adds ~3× compute."
    },
    {
      n: "02", title: "Equivariant attention",
      body: "Replace CFConv with EGNN or PaiNN. More expressive, more parameters, less mature in production codepaths. Punted to v2."
    },
    {
      n: "03", title: "Ligand-only baseline",
      body: "A protein-free baseline (ligand fingerprints + MLP) would establish what fraction of the signal lives in the small molecule alone. Quick to add, fair comparison."
    },
  ];
  return (
    <div style={{ display: "grid", gap: 12, marginTop: 16 }}>
      {items.map((it) => (
        <div key={it.n} className="reveal" style={{ display: "grid", gridTemplateColumns: "60px 1fr", gap: 20, padding: "20px 24px", border: "var(--border-1)", borderRadius: 12, background: "var(--surface-1)" }}>
          <div className="font-mono text-3" style={{ fontSize: 16 }}>{it.n}</div>
          <div>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>{it.title}</div>
            <div className="text-sm text-2" style={{ textWrap: "pretty" }}>{it.body}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ===========================================================
   METHODOLOGY
   =========================================================== */
function PageMethodology() {
  useReveal();
  return (
    <article>
      <ContentHeader
        chip="Methodology"
        title={<>How we trained, evaluated, and what we wrote down <em className="italic" style={{ color: "var(--accent)" }}>so others can rerun it.</em></>}
        sub="Dataset construction, splits, training procedure, ablations. Every metric in the results page traces to a config and a seed."
      />

      <Section paddingY="md">
        <SubsectionTitle eyebrow="01" title="Dataset"/>
        <Prose>
          Our training set is built from the intersection of two public resources. MISATO supplies 100-frame molecular-dynamics trajectories at 10&thinsp;ps spacing for 16,972 protein–ligand complexes. PDBbind+ supplies the experimental log K labels. The intersection — complexes present in both — is the working set.
        </Prose>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginTop: 24 }} className="datas-grid">
          <DataPanel label="Complexes" value="16,972" sub="MISATO ∩ PDBbind+"/>
          <DataPanel label="Frames per complex" value="100" sub="10 ps spacing"/>
          <DataPanel label="Total conformations" value="1,697,200" sub="seen during training"/>
        </div>
      </Section>

      <Section paddingY="md">
        <SubsectionTitle eyebrow="02" title="Splits"/>
        <Prose>
          Two splits, computed once and frozen.
        </Prose>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 16 }} className="splits-grid">
          <SplitCard
            id="random_logk"
            n="1,579 test"
            body="Stratified by binding-affinity bucket so the test set covers the same affinity range as training. The in-distribution split."
          />
          <SplitCard
            id="similarity"
            n="1,565 test"
            body="Test families never appear in training. Protein-sequence-identity threshold of 30%. The out-of-distribution split."
            warn
          />
        </div>
        <Prose>
          We report Pearson, Spearman, R², and RMSE on both splits in <a className="btn-link" href="#/results">Results</a>. Anything that doesn&apos;t generalize between the two splits is suspect.
        </Prose>
      </Section>

      <Section paddingY="md">
        <SubsectionTitle eyebrow="03" title="Training procedure"/>
        <Prose>
          One configuration, one seed, one set of weights checkpointed to GitHub. Reproducibility was a hard requirement: a different machine should be able to rerun any reported number from the config file with one command.
        </Prose>
        <CodeBlock lines={[
          ['# reproduce the headline model', 'c-comment'],
          ['python -m protai.train \\', ''],
          ['    --config configs/multitask_v04.yaml \\', ''],
          ['    --split random_logk \\', ''],
          ['    --seed 42 \\', ''],
          ['    --output runs/multitask_v04', ''],
          ['', ''],
          ['# expected: Pearson 0.414 ± 0.006', 'c-comment'],
        ]}/>
        <Prose>
          One frame per complex per epoch — randomly sampled — keeps the per-epoch cost reasonable while exposing the model to the full trajectory over many epochs.
        </Prose>
      </Section>

      <Section paddingY="md">
        <SubsectionTitle eyebrow="04" title="Ablations"/>
        <Prose>
          The four configurations we report in the paper. Each is a full training run from scratch; differences come from the data and loss, not from architecture.
        </Prose>
        <AblationTable/>
      </Section>

      <Section paddingY="md">
        <SubsectionTitle eyebrow="05" title="What we will and will not claim"/>
        <ClaimsList/>
      </Section>
    </article>
  );
}

function DataPanel({ label, value, sub }) {
  return (
    <div className="card reveal" style={{ padding: 24 }}>
      <div className="text-xs uppercase tracking-wide text-3" style={{ marginBottom: 12 }}>{label}</div>
      <div className="font-mono tabular-nums" style={{ fontSize: "2rem", lineHeight: 1, letterSpacing: "-0.02em" }}>{value}</div>
      <div className="text-xs text-3 font-mono" style={{ marginTop: 8 }}>{sub}</div>
    </div>
  );
}

function SplitCard({ id, n, body, warn }) {
  return (
    <div className="card reveal" style={{ padding: 24, borderColor: warn ? "rgba(245, 194, 107, 0.3)" : undefined }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <span className="font-mono" style={{ fontSize: 14, fontWeight: 600 }}>{id}</span>
        <span className="chip" style={{ background: warn ? "rgba(245, 194, 107, 0.12)" : "var(--surface-2)", color: warn ? "var(--warning)" : "var(--text-2)", borderColor: "transparent" }}>{warn ? "out of distribution" : "in distribution"}</span>
      </div>
      <div className="font-mono text-xs text-3 tabular-nums" style={{ marginBottom: 12 }}>n = {n}</div>
      <p className="text-sm text-2" style={{ textWrap: "pretty" }}>{body}</p>
    </div>
  );
}

function AblationTable() {
  const rows = [
    ["Static · crystal frame",        "0.331", "0.301", "1.71", "0.110", "0.311"],
    ["Trajectory · 100 frames",       "0.405", "0.378", "1.66", "0.164", "0.056", "warn"],
    ["+ adaptability features",       "0.408", "0.379", "1.66", "0.166", "0.087", ""],
    ["+ multitask MD-energy head",    "0.414", "0.388", "1.65", "0.170", "0.244", "highlight"],
  ];
  return (
    <div style={{ overflowX: "auto", marginTop: 16 }}>
      <table className="table-booktabs">
        <thead>
          <tr>
            <th>Configuration</th>
            <th className="num">Pearson&nbsp;r</th>
            <th className="num">Spearman&nbsp;ρ</th>
            <th className="num">RMSE</th>
            <th className="num">R²</th>
            <th className="num">cross-eval&nbsp;r</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const cls = r[6] === "highlight" ? "highlight" : "";
            return (
              <tr key={r[0]} className={cls}>
                <td className="text-1">{r[0]}</td>
                <td className="num">{r[1]}</td>
                <td className="num">{r[2]}</td>
                <td className="num">{r[3]}</td>
                <td className="num">{r[4]}</td>
                <td className="num" style={{ color: r[6] === "warn" ? "var(--negative)" : r[6] === "highlight" ? "var(--accent)" : "var(--text-2)" }}>{r[5]}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="text-xs text-3 font-mono" style={{ marginTop: 14 }}>
        Test sets: random_logk · n = 1,579 · all columns except last · similarity · n = 1,565 · last column.
      </div>
    </div>
  );
}

function ClaimsList() {
  const will = [
    "Trajectory training improves in-distribution Pearson by 22.4% over the static baseline.",
    "An MD-energy auxiliary head closes most of the cross-evaluation gap that pure trajectory training opens.",
    "The full pipeline runs reproducibly from a single config with a fixed seed.",
  ];
  const wont = [
    "That ProtAI replaces wet-lab affinity assays for new chemistry. Cross-eval Pearson 0.244 is informative, not deployable.",
    "That trajectory features alone are universally better than static — they trade generalization for accuracy.",
    "That the model has any business making predictions outside MISATO's chemical envelope.",
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginTop: 16 }} className="claims-grid">
      <ClaimBlock title="What we claim" items={will} positive/>
      <ClaimBlock title="What we do not claim" items={wont}/>
    </div>
  );
}

function ClaimBlock({ title, items, positive }) {
  return (
    <div className="card reveal" style={{ padding: 24 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
        <span style={{
          width: 18, height: 18, borderRadius: 999,
          background: positive ? "var(--positive)" : "var(--surface-3)",
          color: positive ? "var(--bg)" : "var(--text-3)",
          display: "grid", placeItems: "center", fontSize: 12, fontWeight: 700,
        }}>{positive ? "✓" : "−"}</span>
        <span className="text-xs uppercase tracking-wide text-2">{title}</span>
      </div>
      <ul style={{ listStyle: "none", padding: 0, display: "grid", gap: 14 }}>
        {items.map((t, i) => (
          <li key={i} className="text-2 text-sm" style={{ paddingLeft: 18, borderLeft: "2px solid var(--surface-3)", textWrap: "pretty" }}>{t}</li>
        ))}
      </ul>
    </div>
  );
}

/* ===========================================================
   RESULTS
   =========================================================== */
function PageResults() {
  useReveal();
  return (
    <article>
      <ContentHeader
        chip="Results · v0.4 multitask"
        title={<>The four tables and five figures, <em className="italic" style={{ color: "var(--accent)" }}>traceable to a config and a seed.</em></>}
        sub="If you cite any number from this page, it should match the entry in the corresponding JSON under runs/."
      />

      <Section paddingY="md">
        <SubsectionTitle eyebrow="Table 1" title="In-distribution metrics · random_logk split"/>
        <Prose>
          Pearson lifts from 0.331 (static) to 0.414 (multitask). RMSE drops 3.5%.
        </Prose>
        <AblationTable/>
      </Section>

      <Section paddingY="md">
        <SubsectionTitle eyebrow="Figure 1" title="Predicted vs. measured · multitask model"/>
        <FigureCard
          src="/figures/fig_predictions.png"
          alt="Scatter plot of predicted vs. measured log K on the random_logk test split."
          caption="Each point is one of the 1,579 test complexes. Diagonal line is y = x. Pearson r = 0.414. The slight regression-to-the-mean is visible at both tails."
        />
      </Section>

      <Section paddingY="md">
        <SubsectionTitle eyebrow="Figure 2" title="Training curves · validation Pearson"/>
        <FigureCard
          src="/figures/fig_training_curves.png"
          alt="Training and validation Pearson over epochs for the multitask model."
          caption="120 epochs, cosine learning-rate decay, single seed (42). Multitask training is slower in the first 30 epochs and overtakes single-task by epoch 60."
        />
      </Section>

      <Section paddingY="md">
        <SubsectionTitle eyebrow="Figure 3" title="Loss curves · primary vs. auxiliary"/>
        <FigureCard
          src="/figures/fig_loss_curves.png"
          alt="Training and validation losses for the affinity head and the MD-energy auxiliary head."
          caption="Both heads continue to improve; neither saturates within the budget. Suggests a longer training schedule could push numbers further."
        />
      </Section>

      <Section paddingY="md">
        <SubsectionTitle eyebrow="Figure 4" title="In-dist vs cross-eval · per configuration"/>
        <FigureCard
          src="/figures/fig_cross_eval.png"
          alt="Bar chart comparing in-distribution and cross-evaluation Pearson for static, trajectory, and multitask configurations."
          caption="Solid bars are in-distribution Pearson on the random_logk test split; hatched bars are cross-evaluation Pearson on MISATO's similarity split. Trajectory shows the largest in-dist→OOD drop; multitask threads the needle by giving up only a little in-dist gain."
        />
      </Section>

      <Section paddingY="md">
        <SubsectionTitle eyebrow="Table 2" title="Cross-evaluation · similarity split"/>
        <Prose>
          The trade-off, in numbers. Static crystal Pearson 0.311 is the best generic baseline on novel protein families. Trajectory alone collapses to 0.056. Multitask recovers to 0.244 — about 78% of the static-baseline correlation, while keeping the in-distribution gain.
        </Prose>
        <CrossEvalTable/>
      </Section>

      <Section paddingY="md">
        <SubsectionTitle eyebrow="Table 3" title="Per-family breakdown · multitask cross-eval"/>
        <FamilyTable/>
      </Section>

      <Section paddingY="md">
        <SubsectionTitle eyebrow="Figure 5" title="Four-metric summary · per configuration"/>
        <FigureCard
          src="/figures/fig_metric_panel.png"
          alt="Four-panel chart showing Pearson, Spearman, RMSE, and MAE for all configurations on both test splits."
          caption="A compact summary of the same configurations across the four headline metrics. RMSE and MAE in -log K units; Pearson and Spearman dimensionless. Higher is better for correlations, lower for errors."
        />
      </Section>

      <Section paddingY="md">
        <SubsectionTitle eyebrow="04" title="Caveats"/>
        <CaveatGrid/>
      </Section>

      <Section paddingY="md">
        <div className="card" style={{ padding: 28, background: "var(--surface-2)" }}>
          <div style={{ display: "flex", alignItems: "flex-start", gap: 20, flexWrap: "wrap", justifyContent: "space-between" }}>
            <div style={{ maxWidth: 480 }}>
              <div className="text-xs uppercase tracking-wide text-3" style={{ marginBottom: 8 }}>Cite this work</div>
              <h3 className="font-display" style={{ fontSize: "1.5rem", lineHeight: 1.2 }}>BibTeX</h3>
            </div>
            <a href="#/reference" className="btn-link">Full reference →</a>
          </div>
          <BibtexBlock/>
        </div>
      </Section>
    </article>
  );
}

function FigureCard({ src, alt, caption }) {
  const [loaded, setLoaded] = useState(false);
  return (
    <figure className="reveal card" style={{ padding: 16, marginTop: 12 }}>
      <div style={{ position: "relative", background: "var(--surface-2)", borderRadius: 8, overflow: "hidden", minHeight: 200 }}>
        {!loaded && (
          <div className="placeholder-img" style={{ position: "absolute", inset: 0 }}>
            loading figure…
          </div>
        )}
        <img
          src={src}
          alt={alt}
          onLoad={() => setLoaded(true)}
          onError={() => setLoaded(true)}
          style={{ width: "100%", display: "block", opacity: loaded ? 1 : 0, transition: "opacity 220ms ease" }}
        />
      </div>
      <figcaption className="text-sm text-2" style={{ marginTop: 12, textWrap: "pretty", paddingLeft: 4 }}>
        {caption}
      </figcaption>
    </figure>
  );
}

function CrossEvalTable() {
  const rows = [
    ["Static · crystal frame",        "0.311", "0.298", "1.83", "0.094"],
    ["Trajectory · 100 frames",       "0.056", "0.061", "2.21", "0.003", "warn"],
    ["+ adaptability features",       "0.087", "0.092", "2.18", "0.008"],
    ["+ multitask MD-energy head",    "0.244", "0.227", "1.94", "0.046", "highlight"],
  ];
  return (
    <div style={{ overflowX: "auto", marginTop: 16 }}>
      <table className="table-booktabs">
        <thead><tr>
          <th>Configuration</th>
          <th className="num">Pearson&nbsp;r</th>
          <th className="num">Spearman&nbsp;ρ</th>
          <th className="num">RMSE</th>
          <th className="num">R²</th>
        </tr></thead>
        <tbody>
          {rows.map((r) => {
            const cls = r[5] === "highlight" ? "highlight" : "";
            return (
              <tr key={r[0]} className={cls}>
                <td className="text-1">{r[0]}</td>
                <td className="num" style={{ color: r[5] === "warn" ? "var(--negative)" : r[5] === "highlight" ? "var(--accent)" : "var(--text-2)" }}>{r[1]}</td>
                <td className="num">{r[2]}</td>
                <td className="num">{r[3]}</td>
                <td className="num">{r[4]}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="text-xs text-3 font-mono" style={{ marginTop: 14 }}>similarity split · 30% sequence-identity threshold · n = 1,565.</div>
    </div>
  );
}

function FamilyTable() {
  const rows = [
    ["Kinase",                 412, "0.311"],
    ["Aspartic protease",      298, "0.402"],
    ["Serine protease",        287, "0.358"],
    ["Carbonic anhydrase",     156, "0.276"],
    ["Nuclear hormone receptor", 124, "0.198"],
    ["Other / unclassified",   288, "0.166"],
  ];
  return (
    <div style={{ overflowX: "auto", marginTop: 16 }}>
      <table className="table-booktabs">
        <thead><tr>
          <th>Protein family</th>
          <th className="num">n test</th>
          <th className="num">Pearson&nbsp;r</th>
        </tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r[0]}>
              <td className="text-1">{r[0]}</td>
              <td className="num text-2">{r[1].toLocaleString()}</td>
              <td className="num">{r[2]}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="text-xs text-3 font-mono" style={{ marginTop: 14 }}>multitask model · similarity split · families ordered by training prevalence.</div>
    </div>
  );
}

function CaveatGrid() {
  const items = [
    { t: "Single seed", b: "All reported numbers use seed = 42. We re-ran the multitask configuration with seeds 7, 17, 99: σ(Pearson) ≈ 0.006 in distribution, ≈ 0.014 cross-eval." },
    { t: "MISATO is the universe", b: "Anything outside the MISATO chemical and structural envelope is extrapolation. Cross-eval is informative but not predictive." },
    { t: "log K, not free energy", b: "We predict the PDBbind log K label directly, not absolute binding free energy. Conversion factors exist but are domain- and assay-specific." },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginTop: 16 }} className="caveats-grid">
      {items.map((c) => (
        <div key={c.t} className="card reveal" style={{ padding: 22 }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>{c.t}</div>
          <p className="text-sm text-2" style={{ textWrap: "pretty" }}>{c.b}</p>
        </div>
      ))}
    </div>
  );
}

function BibtexBlock() {
  const ref = useRef(null);
  const [copied, setCopied] = useState(false);
  const code = `@misc{protai2026,
  title  = {ProtAI: Trajectory-aware protein–ligand binding affinity prediction},
  author = {Shoro, Hassan Ali and Chaudhry, Ibaad Ahmed and Sheikh, Abdullah Kaif},
  year   = {2026},
  school = {FAST NUCES Islamabad},
  note   = {Final-year project. Code: https://github.com/LastPredator/ProtAI}
}`;
  const copy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    });
  };
  return (
    <div style={{ position: "relative", marginTop: 18 }}>
      <pre className="font-mono" style={{
        background: "var(--surface-1)", border: "var(--border-1)", borderRadius: 10,
        padding: "18px 20px", fontSize: 13, lineHeight: 1.5, overflowX: "auto",
        color: "var(--text-1)", whiteSpace: "pre-wrap",
      }} ref={ref}>{code}</pre>
      <button onClick={copy} className="btn btn-secondary btn-sm" style={{ position: "absolute", top: 12, right: 12 }}>
        {copied ? "✓ Copied" : "Copy"}
      </button>
    </div>
  );
}

/* ===========================================================
   REFERENCE
   =========================================================== */
function PageReference() {
  useReveal();
  const refs = [
    {
      k: "MISATO",
      t: "MISATO: machine learning dataset of protein–ligand complexes for structure-based drug discovery",
      authors: "Siebenmorgen et al.",
      year: "2024",
      where: "Nature Computational Science",
      url: "https://zenodo.org/records/7711953",
    },
    {
      k: "PDBbind+",
      t: "PDBbind+: A continuously updated, manually curated resource of binding-affinity data",
      authors: "Wang et al.",
      year: "2024",
      where: "Nucleic Acids Research",
      url: "http://www.pdbbind-plus.org.cn/",
    },
    {
      k: "SchNet",
      t: "SchNet — A continuous-filter convolutional neural network for modeling quantum interactions",
      authors: "Schütt et al.",
      year: "2017",
      where: "NeurIPS",
      url: "https://arxiv.org/abs/1706.08566",
    },
    {
      k: "PyG",
      t: "Fast Graph Representation Learning with PyTorch Geometric",
      authors: "Fey and Lenssen",
      year: "2019",
      where: "ICLR Workshop",
      url: "https://pytorch-geometric.readthedocs.io/",
    },
    {
      k: "3DMol.js",
      t: "3DMol.js: molecular visualization with WebGL",
      authors: "Rego and Koes",
      year: "2015",
      where: "Bioinformatics",
      url: "https://3dmol.csb.pitt.edu/",
    },
  ];
  return (
    <article>
      <ContentHeader
        chip="Reference"
        title={<>The works ProtAI builds on, <em className="italic" style={{ color: "var(--accent)" }}>and the people behind it.</em></>}
        sub="Bibliography, code attribution, license."
      />

      <Section paddingY="md">
        <SubsectionTitle eyebrow="Bibliography"/>
        <div style={{ display: "grid", gap: 4, marginTop: 16 }}>
          {refs.map((r, i) => (
            <a key={r.k} href={r.url} target="_blank" rel="noopener noreferrer" className="reveal" style={{
              display: "grid",
              gridTemplateColumns: "80px 1fr auto",
              gap: 24,
              padding: "20px 0",
              borderTop: "var(--border-1)",
              alignItems: "baseline",
            }}>
              <div className="font-mono text-3 text-xs uppercase tracking-wide">{String(i+1).padStart(2,"0")} · {r.k}</div>
              <div>
                <div className="font-display" style={{ fontSize: "1.125rem", lineHeight: 1.3, marginBottom: 4 }}>{r.t}</div>
                <div className="text-sm text-2">{r.authors} · {r.where} · {r.year}</div>
              </div>
              <span className="text-xs text-3 font-mono" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                visit
                <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M3 3h6v6m0-6L3 9"/></svg>
              </span>
            </a>
          ))}
          <div style={{ borderTop: "var(--border-1)" }}/>
        </div>
      </Section>

      <Section paddingY="md">
        <SubsectionTitle eyebrow="Team" title="The three people behind this"/>
        <TeamGrid/>
      </Section>

      <Section paddingY="md">
        <SubsectionTitle eyebrow="License" title="MIT"/>
        <Prose>
          The ProtAI source code, model weights, and this site are released under the MIT License. The MISATO dataset and PDBbind+ have their own terms; cite them when you use the model on derived data.
        </Prose>
        <Prose>
          Contact: <a className="btn-link" href="mailto:shorohassanali@gmail.com">shorohassanali@gmail.com</a>. Issues and pull requests welcome on <a className="btn-link" href="https://github.com/LastPredator/ProtAI" target="_blank" rel="noopener noreferrer">GitHub</a>.
        </Prose>
      </Section>
    </article>
  );
}

function TeamGrid() {
  // Team roster — names + roll numbers only. Per-person work distribution
  // intentionally omitted; the project credits all three for the work as
  // a whole.
  const team = [
    { n: "Hassan Ali Shoro",     r: "22I-0561", init: "HS" },
    { n: "Ibaad Ahmed Chaudhry", r: "22I-0585", init: "IA" },
    { n: "Abdullah Kaif Sheikh", r: "22I-2142", init: "AK" },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginTop: 16 }} className="team-grid">
      {team.map((m) => (
        <div key={m.r} className="card reveal" style={{ padding: 24 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <div style={{
              width: 44, height: 44, borderRadius: 999, background: "var(--accent-soft)", color: "var(--accent)",
              display: "grid", placeItems: "center", fontFamily: "Instrument Serif, serif", fontSize: 18, fontWeight: 500
            }}>{m.init}</div>
            <div>
              <div style={{ fontWeight: 600 }}>{m.n}</div>
              <div className="text-xs text-3 font-mono tabular-nums">{m.r}</div>
            </div>
          </div>
        </div>
      ))}
      <div className="card reveal" style={{ padding: 24, gridColumn: "1 / -1", background: "var(--surface-2)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 44, height: 44, borderRadius: 999, background: "var(--surface-3)", color: "var(--text-1)",
            display: "grid", placeItems: "center", fontFamily: "Instrument Serif, serif", fontSize: 18, fontWeight: 500
          }}>SS</div>
          <div>
            <div style={{ fontWeight: 600 }}>Mr. Shoaib Saleem Khattak</div>
            <div className="text-xs text-3">Supervisor · Department of Computer Science · FAST NUCES Islamabad</div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ===========================================================
   Shared content primitives
   =========================================================== */
function ContentHeader({ chip, title, sub }) {
  return (
    <header style={{ borderBottom: "var(--border-1)", padding: "72px 0 56px", background: "var(--surface-1)" }}>
      <div className="container">
        <div className="reveal" style={{ maxWidth: 880 }}>
          {chip && <span className="chip" style={{ marginBottom: 18 }}>{chip}</span>}
          <h1 className="font-display" style={{ fontSize: "clamp(2.25rem, 5vw, 4rem)", lineHeight: 1.05, letterSpacing: "-0.025em", textWrap: "balance" }}>
            {title}
          </h1>
          {sub && <p className="text-lg text-2" style={{ marginTop: 20, maxWidth: 660, textWrap: "pretty" }}>{sub}</p>}
        </div>
      </div>
    </header>
  );
}

function SubsectionTitle({ eyebrow, title }) {
  return (
    <div className="reveal" style={{ marginBottom: 24 }}>
      {eyebrow && <div className="text-xs uppercase tracking-wide text-3 font-mono" style={{ marginBottom: 8 }}>{eyebrow}</div>}
      {title && <h2 className="font-display" style={{ fontSize: "clamp(1.5rem, 2.6vw, 2.25rem)", lineHeight: 1.15, letterSpacing: "-0.015em", textWrap: "balance" }}>{title}</h2>}
    </div>
  );
}

function Prose({ children }) {
  return (
    <p className="text-lg text-2 reveal" style={{ maxWidth: 700, marginTop: 16, lineHeight: 1.7, textWrap: "pretty" }}>{children}</p>
  );
}

function CodeBlock({ lines }) {
  return (
    <pre className="font-mono reveal" style={{
      background: "var(--surface-2)",
      border: "var(--border-1)",
      borderRadius: 10,
      padding: "18px 20px",
      fontSize: 13,
      lineHeight: 1.6,
      overflowX: "auto",
      marginTop: 20,
      maxWidth: 760,
    }}>
      {lines.map(([t, cls], i) => (
        <div key={i} style={{ color: cls === "c-comment" ? "var(--text-3)" : "var(--text-1)" }}>{t || "\u00A0"}</div>
      ))}
    </pre>
  );
}

function Callout({ title, body }) {
  return (
    <aside className="reveal" style={{
      marginTop: 28,
      padding: "20px 24px",
      borderLeft: "2px solid var(--accent)",
      background: "var(--accent-soft)",
      borderRadius: "0 8px 8px 0",
      maxWidth: 720,
    }}>
      <div className="text-xs uppercase tracking-wide" style={{ color: "var(--accent)", marginBottom: 6 }}>{title}</div>
      <div className="text-base text-2" style={{ textWrap: "pretty" }}>{body}</div>
    </aside>
  );
}

Object.assign(window, { PageArchitecture, PageMethodology, PageResults, PageReference });
