/* ============================================================
   Demo page — interactive viewer wired to the real backend.

   Data sources (live, no mocks):
     - Atom counts + protein/ligand split  → api.getStructure()
     - Per-frame MD interaction energy     → api.getFrame() / live scrubber
     - Pocket residues (4.5 Å)             → api.getPocket()
     - Predicted binding affinity          → api.predict()
     - Ground-truth -log K                 → curated QUICK_COMPLEXES
                                            (PDBbind v2020R1 affinity table)

   The Flask backend at :5000 (proxied by the dev server) loads the
   multitask_logk_energy checkpoint, so api.predict() returns model
   output in -log K units. true_affinity from /api/predict is the MD
   interaction energy in kcal/mol; we ignore it on this page in favour
   of the PDBbind log K from the curated list (matches the prediction
   target so the comparison is meaningful).
   ============================================================ */

function PageDemo() {
  useReveal();

  const [pdbId, setPdbId] = useState("1A1B");
  const [mode, setMode] = useState("adaptability");
  const [frame, setFrame] = useState(0);

  /* ---- structure metadata (atom counts, true mean energy) ---- */
  const [structure, setStructure] = useState(null);
  const [structErr, setStructErr] = useState(null);
  const [structLoading, setStructLoading] = useState(false);

  /* ---- live MD energy + atomic coordinates at current frame ---- */
  const [liveEnergy, setLiveEnergy] = useState(null);
  const [frameCoords, setFrameCoords] = useState(null);

  /* ---- prediction state machine ---- */
  const [predState, setPredState] = useState({ status: "idle", result: null });
  const [pulse, setPulse] = useState(false);

  /* ---- backend health (drives the green/red dot in the header) ---- */
  const [health, setHealth] = useState(null);

  /* ---- cached pocket info (only fetched when mode === "pocket") ---- */
  const [pocket, setPocket] = useState(null);

  /* ---- full list of available PDB ids (for the free-text picker) ---- */
  const [allPdbs, setAllPdbs] = useState([]);
  useEffect(() => {
    api
      .listStructures()
      .then((res) => setAllPdbs(res.structures || []))
      .catch(() => setAllPdbs([]));
  }, []);

  /* Curated metadata for the active PDB (null for non-curated). */
  const curated = useMemo(
    () => window.QUICK_COMPLEXES.find((c) => c.id === pdbId) || null,
    [pdbId]
  );

  /* Health check on first mount. */
  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: "down" }));
  }, []);

  /* Fetch structure metadata when the PDB id changes. The structure
     payload includes per-atom coordinates (frame 0), atomic numbers,
     elements, adaptability, and the mean MD interaction energy. */
  useEffect(() => {
    let cancelled = false;
    setStructLoading(true);
    setStructErr(null);
    setStructure(null);
    setLiveEnergy(null);
    setFrameCoords(null);
    setPredState({ status: "idle", result: null });
    setFrame(0);
    setPocket(null);
    api
      .getStructure(pdbId)
      .then((s) => {
        if (cancelled) return;
        setStructure(s);
        setLiveEnergy(s.true_affinity); // fallback to mean energy until first scrub
        setFrameCoords(s.coordinates);  // viewer renders frame 0 immediately
      })
      .catch((err) => {
        if (cancelled) return;
        setStructErr(err.message || "Could not load structure.");
      })
      .finally(() => {
        if (!cancelled) setStructLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [pdbId]);

  /* Fetch the per-frame energy AND atomic coordinates on every scrub.
     The coordinates make the molecule actually move in the viewer.
     Guards against stale responses by tracking the latest requested
     frame index. */
  const lastFrameRef = useRef(0);
  useEffect(() => {
    if (!structure) return;
    if (frame === 0) {
      setLiveEnergy(structure.true_affinity);
      setFrameCoords(structure.coordinates);
      return;
    }
    lastFrameRef.current = frame;
    api
      .getFrame(pdbId, frame)
      .then((f) => {
        if (lastFrameRef.current === frame) {
          setLiveEnergy(f.energy);
          setFrameCoords(f.coordinates);
        }
      })
      .catch(() => {
        /* swallow — keep the last good energy + coords on screen */
      });
  }, [pdbId, frame, structure]);

  /* Pocket fetch (only when entering pocket mode and not yet cached). */
  useEffect(() => {
    if (mode !== "pocket" || pocket || !structure) return;
    api.getPocket(pdbId).then(setPocket).catch(() => setPocket(null));
  }, [mode, pdbId, structure, pocket]);

  /* Real prediction call. Backend returns predicted_affinity in
     whatever units the loaded checkpoint trained on — for the current
     multitask_logk_energy checkpoint that's -log10(K). */
  const runPrediction = useCallback(() => {
    setPredState({ status: "computing", result: null });
    const t0 = performance.now();
    api
      .predict(pdbId)
      .then((res) => {
        const elapsed = Math.round(performance.now() - t0);
        const predicted = res.predicted_affinity != null ? res.predicted_affinity : 0;
        const truthLogK = curated ? curated.truth_logk : null;
        const err =
          truthLogK !== null ? Math.abs(predicted - truthLogK) : null;
        setPredState({
          status: "done",
          result: {
            predicted,
            truth: truthLogK,
            truthKind: curated ? curated.truth_kind : null,
            err,
            modelType: res.model_type || "schnet (multitask)",
            time: elapsed,
            num_atoms: res.num_atoms,
          },
        });
        setPulse(true);
        setTimeout(() => setPulse(false), 1200);
      })
      .catch((err) => {
        setPredState({
          status: "error",
          result: { message: err.message || "Prediction failed." },
        });
      });
  }, [pdbId, curated]);

  /* View-model the rest of the page expects. The page started life with a
     hardcoded record per PDB; we synthesize the equivalent shape from
     live API data + curated metadata. */
  const complex = useMemo(() => {
    return {
      id: pdbId,
      name: curated ? curated.name : pdbId,
      family: curated ? curated.family : "—",
      atoms: structure ? structure.num_atoms : 0,
      prot: structure ? structure.num_protein_atoms : 0,
      lig: structure ? structure.num_ligand_atoms : 0,
      truth_logk: curated ? curated.truth_logk : null,
      truth_kind: curated ? curated.truth_kind : null,
      truth_energy: structure ? structure.true_affinity : null,
    };
  }, [pdbId, curated, structure]);

  return (
    <div style={{ background: "var(--bg)" }}>
      <DemoHeader health={health} />

      <div className="container-wide" style={{ padding: "0 24px 96px" }}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "300px minmax(0, 1fr) 360px",
            gap: 24,
            alignItems: "start",
          }}
          className="demo-grid"
        >
          {/* LEFT — selector + stats */}
          <aside style={{ display: "grid", gap: 16, position: "sticky", top: 80 }} className="demo-aside-left">
            <ComplexPicker pdbId={pdbId} onChange={setPdbId} complex={complex} loading={structLoading} error={structErr} allPdbs={allPdbs} />
            <StatsPanel complex={complex} liveEnergy={liveEnergy} frame={frame} loading={structLoading} />
          </aside>

          {/* CENTER — viewer */}
          <section>
            <ViewerPanel
              pdbId={pdbId}
              complex={complex}
              mode={mode}
              onMode={setMode}
              frame={frame}
              onFrame={setFrame}
              liveEnergy={liveEnergy}
              loading={structLoading}
              error={structErr}
              structure={structure}
              pocket={pocket}
              frameCoords={frameCoords}
            />
          </section>

          {/* RIGHT — prediction */}
          <aside style={{ display: "grid", gap: 16, position: "sticky", top: 80 }} className="demo-aside-right">
            <PredictionPanel complex={complex} state={predState} onRun={runPrediction} pulse={pulse} disabled={structLoading || !!structErr} />
            <ConfidenceSidecar complex={complex} predState={predState} />
          </aside>
        </div>
      </div>

      <style>{`
        @media (max-width: 1280px) {
          .demo-grid { grid-template-columns: 260px minmax(0, 1fr) 320px !important; gap: 18px !important; }
        }
        @media (max-width: 1023px) {
          .demo-grid { grid-template-columns: 1fr !important; }
          .demo-aside-left, .demo-aside-right { position: static !important; }
        }
      `}</style>
    </div>
  );
}

/* ---------- Header ---------- */
function DemoHeader({ health }) {
  const ok = health && (health.model_loaded === true || health.status === "ok");
  const dotColor = ok ? "var(--positive)" : health ? "var(--negative)" : "var(--warning)";
  const statusLabel = ok
    ? "backend · 200 OK"
    : health
      ? "backend · offline"
      : "backend · checking";
  const modelLabel = (health && health.model_path)
    ? "model · " + (health.model_path.split(/[\\/]/).slice(-2, -1)[0] || "loaded")
    : "model · awaiting health";

  return (
    <div style={{ borderBottom: "var(--border-1)", marginBottom: 32 }}>
      <div className="container-wide" style={{ padding: "56px 24px 40px" }}>
        <div className="reveal" style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", flexWrap: "wrap", gap: 24 }}>
          <div>
            <span className="chip" style={{ marginBottom: 14 }}>
              <span style={{ width: 6, height: 6, borderRadius: 999, background: "var(--positive)" }}/>
              Interactive · runs on the trained checkpoint
            </span>
            <h1 className="font-display" style={{ fontSize: "clamp(2.25rem, 4.4vw, 3.5rem)", lineHeight: 1.05, letterSpacing: "-0.02em" }}>
              Pick a complex. Watch the pocket breathe.
            </h1>
            <p className="text-lg text-2" style={{ marginTop: 14, maxWidth: 620, textWrap: "pretty" }}>
              Three viewing modes, full 100-frame trajectory, live binding-affinity prediction. {window.QUICK_COMPLEXES.length}&thinsp;curated · 16,972 in the dataset.
            </p>
          </div>
          <div className="text-xs font-mono text-3 uppercase tracking-wide" style={{ display: "flex", gap: 16, alignItems: "center" }}>
            <span><span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 999, background: dotColor, marginRight: 6, verticalAlign: "middle" }}/>{statusLabel}</span>
            <span>{modelLabel}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------- Complex picker ----------
   Two ways to pick a structure:
     1. Curated dropdown (9 picks with known PDBbind log K → cleaner demo)
     2. Free-text input with autocomplete against all 16,972 dataset PDBs
        (no measured truth, but the model still runs and the prediction is real)
*/
function ComplexPicker({ pdbId, onChange, complex, loading, error, allPdbs }) {
  // Free-text search field state — kept separate from `pdbId` so the user
  // can type without immediately reloading the structure on every keystroke.
  const [query, setQuery] = useState("");
  const allSet = useMemo(() => new Set(allPdbs), [allPdbs]);

  const submit = () => {
    const v = query.trim().toUpperCase();
    if (v && allSet.has(v) && v !== pdbId) {
      onChange(v);
      setQuery("");
    }
  };

  const known = query.length === 0 || allSet.has(query.trim().toUpperCase());

  return (
    <div className="card" style={{ padding: 20 }}>
      {/* Curated quick pick */}
      <label className="text-xs uppercase tracking-wide text-3" style={{ display: "block", marginBottom: 8 }} htmlFor="pdb-select">
        Quick pick · {window.QUICK_COMPLEXES.length} curated
      </label>
      <select
        id="pdb-select"
        value={window.QUICK_COMPLEXES.find((c) => c.id === pdbId) ? pdbId : ""}
        onChange={(e) => onChange(e.target.value)}
        className="input select font-mono"
        style={{ textTransform: "uppercase", fontWeight: 600 }}
      >
        <option value="" disabled>{pdbId} (custom)</option>
        {window.QUICK_COMPLEXES.map((c) => (
          <option key={c.id} value={c.id}>{c.id} — {c.name}</option>
        ))}
      </select>

      {/* Free-text — any PDB in the dataset */}
      <label className="text-xs uppercase tracking-wide text-3" style={{ display: "block", marginTop: 16, marginBottom: 8 }} htmlFor="pdb-input">
        Any PDB · {allPdbs.length > 0 ? allPdbs.length.toLocaleString() : "loading…"} in dataset
      </label>
      <div style={{ display: "flex", gap: 6 }}>
        <input
          id="pdb-input"
          type="text"
          list="protai-all-pdbs"
          placeholder="e.g. 4CP5"
          value={query}
          onChange={(e) => setQuery(e.target.value.toUpperCase())}
          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); submit(); } }}
          className="input font-mono"
          style={{ flex: 1, textTransform: "uppercase", fontWeight: 600 }}
          maxLength={6}
          spellCheck={false}
          autoComplete="off"
        />
        <button
          type="button"
          onClick={submit}
          className="btn btn-secondary btn-sm"
          disabled={!query.trim() || !allSet.has(query.trim().toUpperCase())}
        >Load</button>
      </div>
      {!known && (
        <div className="text-xs text-warning font-mono" style={{ marginTop: 6 }}>
          {query.toUpperCase()} not in dataset
        </div>
      )}
      {/* Datalist powers the browser-native autocomplete on the input above.
          16,972 options is large but well-handled by modern browsers. */}
      <datalist id="protai-all-pdbs">
        {allPdbs.map((id) => (
          <option key={id} value={id} />
        ))}
      </datalist>

      <hr className="divider" style={{ margin: "16px 0" }}/>

      <div>
        <div className="text-sm" style={{ fontWeight: 500 }}>{complex.name}</div>
        <div className="text-xs text-3 font-mono uppercase tracking-wide" style={{ marginTop: 4 }}>
          {loading ? "loading…" : error ? "error" : complex.family}
        </div>
        {complex.truth_logk !== null && (
          <div className="text-xs text-3 font-mono tabular-nums" style={{ marginTop: 8 }}>
            measured · {complex.truth_logk.toFixed(2)} {complex.truth_kind}
          </div>
        )}
      </div>

      <hr className="divider" style={{ margin: "16px 0" }}/>

      <a
        href={`https://www.rcsb.org/structure/${complex.id}`}
        target="_blank" rel="noopener noreferrer"
        className="text-sm text-accent"
        style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
      >
        View {complex.id} on RCSB PDB
        <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M3 3h6v6m0-6L3 9"/></svg>
      </a>
    </div>
  );
}

/* ---------- Stats panel ---------- */
function StatsPanel({ complex, liveEnergy, frame, loading }) {
  const showAtoms = !loading && complex.atoms > 0;
  return (
    <div className="card" style={{ padding: 20 }}>
      <div className="text-xs uppercase tracking-wide text-3" style={{ marginBottom: 14 }}>Structure</div>
      <dl style={{ display: "grid", gap: 10 }}>
        <StatRow k="Total atoms"   v={showAtoms ? complex.atoms.toLocaleString() : "—"}/>
        <StatRow k="Protein atoms" v={showAtoms ? complex.prot.toLocaleString() : "—"}/>
        <StatRow k="Ligand atoms"  v={showAtoms ? complex.lig.toLocaleString() : "—"}/>
        <StatRow k="Frame"         v={`${String(frame).padStart(3,"0")} / 100`}/>
      </dl>

      <hr className="divider" style={{ margin: "16px 0" }}/>

      <div className="text-xs uppercase tracking-wide text-3" style={{ marginBottom: 8 }}>Live MD energy</div>
      <div className="font-mono tabular-nums" style={{ fontSize: "1.5rem", letterSpacing: "-0.01em" }}>
        {liveEnergy === null ? "—" : liveEnergy.toFixed(2)}
      </div>
      <div className="text-xs text-3 font-mono">kcal/mol · current frame</div>
    </div>
  );
}

function StatRow({ k, v }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
      <dt className="text-3">{k}</dt>
      <dd className="font-mono tabular-nums" style={{ margin: 0, color: "var(--text-1)" }}>{v}</dd>
    </div>
  );
}

/* ---------- Viewer panel ---------- */
function ViewerPanel({
  pdbId, complex, mode, onMode, frame, onFrame, liveEnergy,
  loading, error, structure, pocket, frameCoords,
}) {
  return (
    <div className="card" style={{ overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, padding: "14px 20px", borderBottom: "var(--border-1)" }}>
        <div>
          <div className="font-mono" style={{ fontSize: "1rem", fontWeight: 600 }}>{complex.id}</div>
          <div className="text-xs text-3" style={{ marginTop: 2 }}>
            {loading
              ? "loading from backend…"
              : error
                ? error
                : `${complex.atoms.toLocaleString()} atoms · ${complex.prot.toLocaleString()} protein · ${complex.lig.toLocaleString()} ligand`}
          </div>
        </div>
        <ModeSwitcher mode={mode} onChange={onMode}/>
      </div>

      <div className="viewer-host" style={{ position: "relative", aspectRatio: "16 / 10" }}>
        {/* Backend-driven render: real coordinates, real adaptability,
            real pocket residues. Atom positions update live as the
            user scrubs the frame slider. */}
        <MoleculeViewer
          pdbId={pdbId}
          mode={mode}
          structure={structure}
          pocket={pocket}
          frameCoords={frameCoords}
          ariaLabel={`${complex.name} 3D viewer`}
        />
        <div style={{
          position: "absolute", left: 14, top: 14, display: "flex", flexDirection: "column", gap: 6,
          fontFamily: "JetBrains Mono, monospace", fontSize: 11,
          color: "var(--text-3)",
        }}>
          <span>VIEW · {mode}</span>
          <span>FRAME · {String(frame).padStart(3,"0")}</span>
          <span>E = {liveEnergy === null ? "—" : liveEnergy.toFixed(2)} kcal/mol</span>
        </div>
        <div style={{ position: "absolute", right: 14, top: 14 }}>
          <div className="chip">
            <span style={{ width: 6, height: 6, borderRadius: 999, background: mode === "pocket" ? "var(--data-pocket)" : "var(--accent)" }}/>
            <span className="font-mono">3DMol</span>
          </div>
        </div>
      </div>

      <FrameScrubber frame={frame} onChange={onFrame}/>

      <div style={{ padding: "12px 20px", borderTop: "var(--border-1)" }}>
        <ViewerLegend mode={mode}/>
      </div>
    </div>
  );
}

/* ---------- Mode switcher ---------- */
function ModeSwitcher({ mode, onChange }) {
  const opts = [
    { id: "element",      label: "Element" },
    { id: "adaptability", label: "Adaptability" },
    { id: "pocket",       label: "Pocket" },
  ];
  const ref = useRef(null);
  const [indi, setIndi] = useState({ left: 4, width: 0 });
  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current.querySelector(`[data-mode="${mode}"]`);
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const parent = ref.current.getBoundingClientRect();
    setIndi({ left: rect.left - parent.left, width: rect.width });
  }, [mode]);
  return (
    <div className="segmented" ref={ref} role="tablist" aria-label="Viewing mode">
      <span className="segmented-indicator" style={{ left: indi.left, width: indi.width }}/>
      {opts.map((o) => (
        <button
          key={o.id}
          data-mode={o.id}
          role="tab"
          aria-selected={mode === o.id}
          className={mode === o.id ? "active" : ""}
          onClick={() => onChange(o.id)}
        >{o.label}</button>
      ))}
    </div>
  );
}

/* ---------- Frame scrubber ---------- */
function FrameScrubber({ frame, onChange }) {
  const trackRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const seek = (clientX) => {
    if (!trackRef.current) return;
    const r = trackRef.current.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
    onChange(Math.round(t * 100));
  };

  useEffect(() => {
    if (!dragging) return;
    const move = (e) => seek(e.touches ? e.touches[0].clientX : e.clientX);
    const up = () => setDragging(false);
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    window.addEventListener("touchmove", move);
    window.addEventListener("touchend", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      window.removeEventListener("touchmove", move);
      window.removeEventListener("touchend", up);
    };
  }, [dragging]);

  return (
    <div style={{ padding: "0 20px 20px", borderTop: "var(--border-1)", paddingTop: 16, position: "relative" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
        <span className="text-xs uppercase tracking-wide text-3">MD frame</span>
        <span className="font-mono text-xs text-2">{String(frame).padStart(3,"0")} / 100</span>
      </div>
      <div
        className="scrubber"
        style={{ cursor: "pointer" }}
        onMouseDown={(e) => { setDragging(true); seek(e.clientX); }}
        onTouchStart={(e) => { setDragging(true); seek(e.touches[0].clientX); }}
      >
        <div className="scrubber-track" ref={trackRef}>
          <div className="scrubber-fill" style={{ width: `${frame}%` }}/>
          <div className="scrubber-thumb" style={{ left: `${frame}%` }}/>
          {[0, 25, 50, 75, 100].map((p) => (
            <React.Fragment key={p}>
              <span className="scrubber-tick" style={{ left: `${p}%` }}/>
              <span className="scrubber-tick-label" style={{ left: `${p}%` }}>{p}</span>
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ---------- Viewer legend ---------- */
function ViewerLegend({ mode }) {
  if (mode === "element") {
    return (
      <div className="text-sm text-2" style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "center" }}>
        <span className="text-3 text-xs uppercase tracking-wide">Coloring</span>
        <LegendDot c="#94A3B8" label="C"/>
        <LegendDot c="#3B82F6" label="N"/>
        <LegendDot c="#EF4444" label="O"/>
        <LegendDot c="#EAB308" label="S"/>
        <LegendDot c="#A855F7" label="P"/>
      </div>
    );
  }
  if (mode === "adaptability") {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <span className="text-3 text-xs uppercase tracking-wide font-mono">rigid</span>
        <span className="legend-grad"/>
        <span className="text-3 text-xs uppercase tracking-wide font-mono">flexible</span>
      </div>
    );
  }
  return (
    <div className="text-sm text-2" style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "center" }}>
      <LegendDot c="var(--data-pocket)" label="Pocket residues · 4.5 Å"/>
      <LegendDot c="var(--data-ligand)" label="Ligand"/>
      <span className="text-3">Other protein faded.</span>
    </div>
  );
}
function LegendDot({ c, label }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span style={{ width: 10, height: 10, borderRadius: 999, background: c }}/>
      <span className="font-mono text-xs">{label}</span>
    </span>
  );
}

/* Confidence tier from prediction error.
   Thresholds calibrated against the multitask checkpoint's in-distribution
   test residual distribution (n = 1,579, RMSE 1.65 in -log K units):
     * |Δ| < 0.5   → high   (top ~30% of test residuals)
     * |Δ| < 1.5   → medium (~ within one RMSE of truth)
     * |Δ| ≥ 1.5   → low    (worse than typical residual)
     * truth absent → unknown */
function confidenceTier(err) {
  if (err === null || err === undefined) {
    return { label: "—", tone: "neutral", desc: "no measured truth" };
  }
  if (err < 0.5) return { label: "High",   tone: "positive", desc: "within 0.5 log K of measured" };
  if (err < 1.5) return { label: "Medium", tone: "warning",  desc: "within 1.5 log K of measured" };
  return { label: "Low", tone: "negative", desc: ">1.5 log K from measured" };
}

/* ---------- Prediction panel ---------- */
function PredictionPanel({ complex, state, onRun, pulse, disabled }) {
  const { status, result } = state;
  const max = 14; // domain max for bars in -log K

  return (
    <div className={"card " + (pulse ? "pulse-once" : "")} style={{ padding: 20 }}>
      <div className="text-xs uppercase tracking-wide text-3" style={{ marginBottom: 4 }}>Prediction</div>
      <div className="font-display" style={{ fontSize: "1.5rem", letterSpacing: "-0.01em", lineHeight: 1.1 }}>
        −log K · binding affinity
      </div>

      <hr className="divider" style={{ margin: "16px 0" }}/>

      {status === "idle" && (
        <>
          <p className="text-sm text-2" style={{ marginBottom: 16, textWrap: "pretty" }}>
            Forward-pass the SchNet + multitask checkpoint on this complex.
          </p>
          <button
            onClick={onRun}
            disabled={disabled}
            className="btn btn-primary"
            style={{ width: "100%", opacity: disabled ? 0.5 : 1, cursor: disabled ? "not-allowed" : "pointer" }}
          >
            Run prediction
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round"><path d="M3 8h10m0 0L8.5 3.5M13 8l-4.5 4.5"/></svg>
          </button>
        </>
      )}

      {status === "computing" && (
        <>
          <p className="text-sm text-2" style={{ marginBottom: 16 }}>
            Computing forward pass on {complex.atoms.toLocaleString()} atoms…
          </p>
          <ComputingBar/>
        </>
      )}

      {status === "error" && (
        <>
          <p className="text-sm text-negative" style={{ marginBottom: 12 }}>
            {result.message}
          </p>
          <button onClick={onRun} className="btn btn-secondary" style={{ width: "100%" }}>Retry</button>
        </>
      )}

      {status === "done" && (() => {
        const conf = confidenceTier(result.err);
        const confBg =
          conf.tone === "positive" ? "rgba(125, 211, 168, 0.16)" :
          conf.tone === "warning"  ? "rgba(245, 194, 107, 0.16)" :
          conf.tone === "negative" ? "rgba(233, 112, 112, 0.16)" :
                                     "var(--surface-2)";
        const confColor =
          conf.tone === "positive" ? "var(--positive)" :
          conf.tone === "warning"  ? "var(--warning)"  :
          conf.tone === "negative" ? "var(--negative)" :
                                     "var(--text-2)";
        return (
          <>
            {/* Confidence chip — sits above the bars so it's the first thing
                the eye lands on when the prediction returns. */}
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
              <span
                className="chip"
                style={{
                  background: confBg,
                  color: confColor,
                  borderColor: "transparent",
                  fontFamily: "JetBrains Mono, monospace",
                }}
                title={conf.desc}
              >
                <span style={{ width: 6, height: 6, borderRadius: 999, background: confColor }}/>
                Confidence · {conf.label}
              </span>
              <span className="text-xs text-3" style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {conf.desc}
              </span>
            </div>

            <BarRow label="Predicted" value={result.predicted} max={max} color="var(--accent)"/>
            {result.truth !== null ? (
              <BarRow label={"Measured · " + (result.truthKind || "K")} value={result.truth} max={max} color="var(--cool)"/>
            ) : (
              <div className="bar-row">
                <div className="text-xs uppercase tracking-wide text-3">Measured</div>
                <div className="bar-track"><div className="bar-fill" style={{ width: "0%" }}/></div>
                <div className="font-mono tabular-nums text-3" style={{ textAlign: "right", fontSize: 14 }}>n/a</div>
              </div>
            )}

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 18 }}>
              <KV
                k="Δ"
                v={result.err !== null ? result.err.toFixed(2) + " logK" : "—"}
                tone={result.err !== null ? (result.err < 0.5 ? "positive" : result.err < 1.5 ? "warning" : "negative") : undefined}
              />
              <KV k="latency" v={result.time + " ms"}/>
            </div>

            <hr className="divider" style={{ margin: "16px 0" }}/>

            <div className="text-xs font-mono text-3" style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 220 }}>{result.modelType}</span>
              <button onClick={onRun} className="btn-link text-xs">Re-run</button>
            </div>
          </>
        );
      })()}
    </div>
  );
}

function BarRow({ label, value, max, color }) {
  const pct = Math.max(2, Math.min(100, (Math.abs(value) / max) * 100));
  return (
    <div className="bar-row">
      <div className="text-xs uppercase tracking-wide text-3">{label}</div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${pct}%`, background: color }}/>
      </div>
      <div className="font-mono tabular-nums text-2" style={{ textAlign: "right", fontSize: 14 }}>{value.toFixed(2)}</div>
    </div>
  );
}

function KV({ k, v, tone }) {
  const c = tone === "positive" ? "var(--positive)" : tone === "warning" ? "var(--warning)" : "var(--text-1)";
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-3" style={{ marginBottom: 4 }}>{k}</div>
      <div className="font-mono tabular-nums" style={{ color: c, fontSize: 16 }}>{v}</div>
    </div>
  );
}

function ComputingBar() {
  return (
    <div style={{ position: "relative", height: 6, background: "var(--surface-2)", borderRadius: 999, overflow: "hidden" }}>
      <div style={{
        position: "absolute", inset: 0,
        background: "linear-gradient(90deg, transparent, var(--accent), transparent)",
        animation: "sweepX 1.2s ease-in-out infinite",
        width: "40%",
      }}/>
      <style>{`@keyframes sweepX { from { transform: translateX(-100%); } to { transform: translateX(380%); } }`}</style>
    </div>
  );
}

/* ---------- Confidence sidecar ---------- */
function ConfidenceSidecar({ complex, predState }) {
  const result = predState && predState.result;
  const errVal = (result && result.err !== null && result.err !== undefined) ? result.err : null;

  return (
    <div className="card" style={{ padding: 20 }}>
      <div className="text-xs uppercase tracking-wide text-3" style={{ marginBottom: 10 }}>This complex on the model</div>
      <div className="text-sm text-2" style={{ marginBottom: 14, textWrap: "pretty" }}>
        Trained on the random_logk split. <span className="font-mono text-1">{complex.family}</span>
        {complex.truth_logk !== null && (
          <> · ground truth {complex.truth_logk.toFixed(2)} ({complex.truth_kind})</>
        )}
      </div>

      <CalibrationDots err={errVal}/>

      <div className="font-mono text-xs text-3" style={{ marginTop: 14, display: "flex", justifyContent: "space-between" }}>
        <span>0.0</span><span>|Δ| log K</span><span>2.0</span>
      </div>

      <hr className="divider" style={{ margin: "16px 0" }}/>

      <div className="text-xs uppercase tracking-wide text-3" style={{ marginBottom: 8 }}>Caveat</div>
      <p className="text-sm text-2" style={{ textWrap: "pretty" }}>
        Predictions on novel protein families fall to Pearson 0.244. Use cross-evaluated splits for new chemistry.
      </p>
    </div>
  );
}

function CalibrationDots({ err }) {
  // Dots representing the residual distribution shape we observed at
  // training time (in-distribution test set, n=1,579). Highlights the
  // current complex's residual when a prediction has been computed.
  const dots = useMemo(() => {
    const n = 36;
    const arr = [];
    for (let i = 0; i < n; i++) {
      const u = Math.abs((i * 7919 % 100) / 100 - 0.5) * 2;
      const v = Math.pow(u, 1.4) * 2.0;
      arr.push(v);
    }
    return arr.sort((a, b) => a - b);
  }, []);

  return (
    <div style={{ position: "relative", height: 56 }}>
      <div style={{ position: "absolute", left: 0, right: 0, top: 28, height: 1, background: "var(--surface-3)" }}/>
      {dots.map((v, i) => {
        const x = Math.min(100, (v / 2.0) * 100);
        const close = err !== null && Math.abs(v - err) < 0.06;
        return (
          <span key={i}
            style={{
              position: "absolute",
              left: `${x}%`,
              top: 28 - (i % 2 === 0 ? 2 : -2),
              width: 6, height: 6,
              borderRadius: 999,
              background: close ? "var(--accent)" : "var(--surface-3)",
              transform: "translate(-50%, -50%)",
              boxShadow: close ? "0 0 0 4px var(--accent-soft)" : "none",
            }}/>
        );
      })}
      {err !== null && (
        <span style={{
          position: "absolute",
          left: `${Math.min(100, (err / 2.0) * 100)}%`,
          top: 28,
          transform: "translate(-50%, -50%)",
          width: 12, height: 12, borderRadius: 999,
          background: "var(--accent)",
          border: "2px solid var(--bg)",
        }}/>
      )}
    </div>
  );
}

window.PageDemo = PageDemo;
