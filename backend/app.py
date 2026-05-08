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
    print("=" * 80)
    print("ProtAI Backend")
    print("=" * 80)
    svc = get_service()
    print(f"  Repo:     {REPO_ROOT}")
    print(f"  Frontend: {frontend.relative_to(REPO_ROOT)}  ({'production build' if is_built else 'dev source — UI served by Astro on :4321'})")
    print(f"  Model:    {svc.model_path if svc.model_path else '(none — train first)'}")
    print(f"  Data:     {svc.data_path}")
    print(f"  Model loaded: {svc.model_loaded}")
    print(f"  Data loaded:  {svc.data_loaded}")
    print("=" * 80)
    print("Listening on http://localhost:5000")
    print("API only (in dev). UI: http://localhost:4321")
    print("=" * 80)
    app.run(debug=True, port=5000)
