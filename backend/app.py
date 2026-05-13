"""ProtAI Flask backend — thin entry point.

All business logic lives in `protai.api.service` and `protai.api.routes`.
This file just constructs the Flask app, registers routes, and prints the
startup banner.

Static-folder logic:
- Dev (Option B): Astro on :4321 owns the UI, this Flask instance only
  serves /api/*. The static_folder still resolves to frontend-new/ for
  completeness but isn't used in normal browsing.
- Production: after `npm run build` the `frontend-new/dist/` directory
  contains a real index.html; this code auto-detects it and Flask serves
  the built site directly. No code changes needed at deploy time.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import protai` resolves.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flask import Flask
from flask_cors import CORS

from protai.api.routes import register_routes
from protai.api.service import get_service


def _resolve_frontend_dir() -> Path:
    """Prefer the built dist (production), fall back to source (dev)."""
    dist = REPO_ROOT / "frontend-new" / "dist"
    if (dist / "index.html").exists():
        return dist
    return REPO_ROOT / "frontend-new"


def create_app() -> Flask:
    frontend = _resolve_frontend_dir()
    app = Flask(
        __name__,
        static_folder=str(frontend),
        static_url_path="",
    )
    CORS(app)
    register_routes(app, frontend_dir=str(frontend))
    return app


app = create_app()


if __name__ == "__main__":
    frontend = _resolve_frontend_dir()
    is_built = frontend.name == "dist"
    svc = get_service()

    # Resolve the run dir (parent of the ckpt) for clearer banner output.
    run_name = svc.model_path.parent.name if svc.model_path else "(none)"
    ckpt_short: object = (
        svc.model_path.relative_to(REPO_ROOT) if svc.model_path else "(none — train first)"
    )
    try:
        data_short: object = svc.data_path.relative_to(REPO_ROOT)
    except ValueError:
        data_short = svc.data_path
    fe_mode = "production build" if is_built else "dev source — UI served by Astro on :4321"

    bar = "=" * 80
    print(bar)
    print("ProtAI Backend")
    print(bar)
    print(f"  Repo       : {REPO_ROOT}")
    print(f"  Frontend   : {frontend.relative_to(REPO_ROOT)}  ({fe_mode})")
    print(f"  Run        : {run_name}")
    print(f"  Checkpoint : {ckpt_short}")
    print(f"  Dataset    : {data_short}")
    print(f"  Device     : {svc.device}")
    print(f"  Model OK   : {svc.model_loaded}")
    print(f"  Data  OK   : {svc.data_loaded}")
    print(bar)
    print("API : http://localhost:5000/api/health")
    print("UI  : http://localhost:4321   (run `npm run dev` in frontend-new/)")
    print(bar)

    if not svc.model_loaded:
        print("[warn] No checkpoint loaded. Predictions will return null.")
    if not svc.data_loaded:
        print("[warn] MD.hdf5 not found at the expected path. /api/structures will 500.")

    app.run(debug=True, port=5000)
