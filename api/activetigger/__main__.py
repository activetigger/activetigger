import argparse
from pathlib import Path
import uvicorn

from activetigger.config import config

if __name__ == "__main__":
    """
    Launch activetigger API
    """

    parser = argparse.ArgumentParser(description="Run pyActiveTigger")

    parser.add_argument(
        "-p",
        "--portapi",
        type=int,
        default=5000,
        help="Port number for the API. Default is 5000.",
    )
    args = parser.parse_args()
    reload = config.mode == "dev"
    uvicorn.run(
        "activetigger.app.main:app",
        host="0.0.0.0",
        port=args.portapi,
        reload=reload,
        reload_excludes=["*/projects/*", "*/checkpoint-*"],
        log_level="info",
        reload_dirs=[str(Path(__file__).parent)] if reload else None,
    )
