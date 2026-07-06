/* ============================================================
   ProtAI backend wrapper — real /api/* calls to the Flask app
   on :5000 (proxied by Vite/Astro dev server).
   ============================================================

   Endpoints (mirrors backend/app.py + protai/api/routes.py):

     GET  /api/health                 → { status, model_loaded, data_loaded,
                                          model_path, data_path }
     GET  /api/structures             → { structures: string[], total: number }
     GET  /api/structure/:pdb_id      → { pdb_id, num_atoms, num_protein_atoms,
                                          num_ligand_atoms, coordinates,
                                          elements, atomic_numbers, adaptability,
                                          true_affinity }
     GET  /api/structure/:pdb_id/frame/:n  → { pdb_id, frame, total_frames,
                                               coordinates, energy }
     GET  /api/pocket/:pdb_id?cutoff=4.5   → { pdb_id, cutoff_angs, pocket_size,
                                               protein_pocket_indices,
                                               ligand_indices }
     POST /api/predict   { pdb_id }   → { pdb_id, predicted_affinity,
                                          true_affinity, error, model_type,
                                          num_atoms }

   The model returns predictions in the trained target's units. For the
   current `multitask_logk_energy` checkpoint that's −log10(K) (PDBbind-style
   affinity), and `true_affinity` from the structure endpoint is the per-frame
   MD interaction energy averaged across the trajectory in kcal/mol. They
   are NOT the same target; the demo prediction panel shows both side-by-side
   labelled clearly. */

const API_BASE = "/api";

async function fetchJSON(url, init) {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

window.api = {
  health() {
    return fetchJSON(`${API_BASE}/health`);
  },

  listStructures() {
    return fetchJSON(`${API_BASE}/structures`);
  },

  getStructure(pdbId) {
    return fetchJSON(`${API_BASE}/structure/${encodeURIComponent(pdbId)}`);
  },

  getFrame(pdbId, frame) {
    return fetchJSON(`${API_BASE}/structure/${encodeURIComponent(pdbId)}/frame/${frame}`);
  },

  getPocket(pdbId, cutoff = 4.5) {
    return fetchJSON(`${API_BASE}/pocket/${encodeURIComponent(pdbId)}?cutoff=${cutoff}`);
  },

  predict(pdbId) {
    return fetchJSON(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pdb_id: pdbId }),
    });
  },
};

/* ============================================================
   Curated quick-pick complexes for the demo dropdown.

   Avoids forcing the user to scroll a 16,972-entry list.
   `truth_logk` is the PDBbind v2020R1 −log10(K) ground truth for that
   complex (extracted from data/MD/affinity.csv on the build pod).
   `truth_kind` is the assay (Kd / Ki / IC50).

   The full backend list (api.listStructures()) is also available; the
   picker can fall back to free-text PDB entry. The truth_logk for any
   non-curated PDB requires a future backend endpoint that exposes
   the affinity table — for now, only curated entries show a measured
   ground truth.
   ============================================================ */
/* Audited 2026-05-15: every entry must exist in BOTH MISATO's MD.hdf5
   AND data/MD/affinity.csv (PDBbind v2020R1). Entries that are only in
   PDBbind silently 404 the demo's structure endpoint, since the model
   has no MD trajectory for them. The previous list included 1A4K and
   1ABF which were PDBbind-only — replaced with 1A42 and 1A86 which
   appear in both datasets. */
window.QUICK_COMPLEXES = [
  { id: "1A1B", name: "HIV-1 protease · cyclic urea",            family: "Aspartic protease",        truth_logk: 6.398, truth_kind: "Kd"   },
  { id: "1A28", name: "Progesterone receptor",                   family: "Nuclear hormone receptor", truth_logk: 8.292, truth_kind: "Ki"   },
  { id: "1A30", name: "HIV-1 protease · ABT-538",                family: "Aspartic protease",        truth_logk: 4.301, truth_kind: "Ki"   },
  { id: "1A42", name: "Carbonic anhydrase II · brinzolamide",    family: "Lyase",                    truth_logk: 9.886, truth_kind: "Kd"   },
  { id: "1A86", name: "Phospholipase A2 · MAFP",                 family: "Hydrolase",                truth_logk: 3.996, truth_kind: "IC50" },
  { id: "1BCU", name: "Trypsin · benzamidine",                   family: "Serine protease",          truth_logk: 3.276, truth_kind: "Kd"   },
  { id: "1HVH", name: "HIV-1 protease · A-77003",                family: "Aspartic protease",        truth_logk: 7.959, truth_kind: "Ki"   },
  { id: "10GS", name: "Glutathione S-transferase",               family: "Transferase",              truth_logk: 6.398, truth_kind: "Ki"   },
  { id: "184L", name: "T4 lysozyme · indole",                    family: "Hydrolase",                truth_logk: 4.721, truth_kind: "Kd"   },
];
