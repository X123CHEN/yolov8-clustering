"""One entry point for all four offline latent-domain experiments."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import latent_dg_config as settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=settings.EXPERIMENTS, help="Optional explicit override of EXPERIMENT")
    parser.add_argument("--results-path", type=Path, help="Fresh experiment directory (not parent)")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true", help="Verify/generate source membership; no GPU or training")
    parser.add_argument("--resolved-config", type=Path, help="Frozen JSON configuration supplied by the suite runner")
    parser.add_argument("--resume", type=Path, help="Existing run directory; use saved config and epoch-boundary state")
    args = parser.parse_args()
    if args.smoke_steps < 0:
        parser.error("smoke-steps must be nonnegative")
    if args.resume and (args.experiment or args.resolved_config or args.results_path or args.smoke_steps or args.prepare_only):
        parser.error("--resume must be used alone")
    if args.resolved_config and args.experiment:
        parser.error("Do not override a frozen configuration")
    if args.resume:
        config = settings.validate(json.loads((args.resume / "run_config.json").read_text())["resolved_config"])
    else:
        config = settings.validate(json.loads(args.resolved_config.read_text())) if args.resolved_config else settings.resolve(args.experiment)
    # Must happen before importing torch or any legacy CUDA modules.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", config["gpu"])
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.chdir(settings.ROOT)
    from utils.latent_membership import ensure_cache
    cache, manifest = ensure_cache(config["membership"])
    print("MEMBERSHIP", cache, "prior=", manifest["prior"], flush=True)
    if args.prepare_only:
        return
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    output = args.resume or args.results_path or Path(config["results_root"]) / f"{config['experiment']}_seed{config['seed']}_{timestamp}"
    from utils.latent_ablation_training import train
    train(config, output.resolve(), cache, smoke_steps=args.smoke_steps, resume=bool(args.resume))


if __name__ == "__main__":
    main()
