
from __future__ import annotations
import argparse
from pathlib import Path

from .train import run_train
from .eval import run_eval
from .benchmark import run_suite


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--benchmark", default="associative_recall")
    t.add_argument("--model", default="gmf")

    e = sub.add_parser("eval")
    e.add_argument("--benchmark", default="associative_recall")
    e.add_argument("--model", default="gmf")
    e.add_argument("--ckpt", default=None)

    b = sub.add_parser("bench")

    args = p.parse_args()
    if args.cmd == "train":
        run_train(benchmark=args.benchmark, model_name=args.model)
    elif args.cmd == "eval":
        run_eval(benchmark=args.benchmark, model_name=args.model, ckpt=args.ckpt)
    elif args.cmd == "bench":
        run_suite()

if __name__ == "__main__":
    main()
