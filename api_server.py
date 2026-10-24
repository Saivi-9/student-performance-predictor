"""
Academic Performance Predictor REST API Server Entry Point.
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Student Academic Performance Predictor REST API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on (default: 8001)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    args = parser.parse_args()

    print(f"[Academic Predictor] Starting REST API & UI Studio on http://{args.host}:{args.port}")
    print(f"[Academic Predictor] Interactive OpenAPI docs at: http://{args.host}:{args.port}/docs")

    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
