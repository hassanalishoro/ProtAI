"""Flask routes — thin layer over `ProtAIService`.

Mounted into `backend/app.py` via `register_routes(app)`. No business logic
lives here; it's all in `service.py` so we can test it without Flask.
"""
from __future__ import annotations

from flask import Flask, jsonify, request, send_from_directory

from .service import get_service


def register_routes(app: Flask, frontend_dir: str = "../frontend") -> None:
    """Attach all ProtAI HTTP routes to the given Flask app."""

    @app.route("/")
    def serve_index():
        return send_from_directory(frontend_dir, "index.html")

    @app.route("/<path:path>")
    def serve_static(path):
        return send_from_directory(frontend_dir, path)

    @app.route("/api/structures", methods=["GET"])
    def get_structures():
        svc = get_service()
        if not svc.data_loaded:
            return jsonify({"error": "Dataset not loaded"}), 500
        ids = svc.list_structures()
        return jsonify({"structures": ids, "total": len(ids)})

    @app.route("/api/structure/<pdb_id>", methods=["GET"])
    def get_structure_data(pdb_id):
        svc = get_service()
        if not svc.data_loaded:
            return jsonify({"error": "Dataset not loaded"}), 500
        try:
            return jsonify(svc.get_structure_info(pdb_id))
        except KeyError:
            return jsonify({"error": f"Structure {pdb_id} not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/predict", methods=["POST"])
    def predict():
        body = request.get_json(silent=True) or {}
        pdb_id = body.get("pdb_id")
        if not pdb_id:
            return jsonify({"error": "pdb_id required"}), 400
        svc = get_service()
        if not svc.data_loaded:
            return jsonify({"error": "Dataset not loaded"}), 500
        try:
            return jsonify(svc.predict_affinity(pdb_id))
        except KeyError:
            return jsonify({"error": f"Structure {pdb_id} not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/structure/<pdb_id>/frame/<int:frame>", methods=["GET"])
    def get_frame(pdb_id, frame):
        svc = get_service()
        if not svc.data_loaded:
            return jsonify({"error": "Dataset not loaded"}), 500
        try:
            return jsonify(svc.get_frame(pdb_id, frame))
        except KeyError:
            return jsonify({"error": f"Structure {pdb_id} not found"}), 404
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/pocket/<pdb_id>", methods=["GET"])
    def analyze_binding_pocket(pdb_id):
        svc = get_service()
        if not svc.data_loaded:
            return jsonify({"error": "Dataset not loaded"}), 500
        try:
            cutoff = float(request.args.get("cutoff", 4.5))
            return jsonify(svc.analyze_pocket(pdb_id, cutoff=cutoff))
        except KeyError:
            return jsonify({"error": f"Structure {pdb_id} not found"}), 404
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/health", methods=["GET"])
    def health():
        svc = get_service()
        return jsonify({
            "status": "ok",
            "model_loaded": svc.model_loaded,
            "data_loaded": svc.data_loaded,
            "model_path": str(svc.model_path) if svc.model_path else None,
            "data_path": str(svc.data_path),
        })
