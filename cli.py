import argparse
import json
import sys

from src import config
from src.evaluate import evaluate, print_report
from src.inference import RootCauseClassifier
from src.summarize import summarize, summarize_with_llm
from src.train import train as run_train


def cmd_train(args):
    pipe, y_true, y_pred = run_train(save=True)
    print_report(y_true, y_pred)
    m = evaluate(y_true, y_pred, save=True)
    print(f"accuracy={m['accuracy']}  f1_macro={m['f1_macro']}  "
          f"precision_macro={m['precision_macro']}  recall_macro={m['recall_macro']}")


def cmd_evaluate(args):
    path = config.OUTPUTS_DIR / "metrics.json"
    if not path.exists():
        sys.exit("No metrics found. Run `python cli.py train` first.")
    m = json.load(open(path))
    print(json.dumps({k: v for k, v in m.items() if k != "per_class"}, indent=2))


def cmd_predict(args):
    clf = RootCauseClassifier()
    if args.text:
        messages = [args.text]
    elif args.file:
        messages = [ln.strip() for ln in open(args.file) if ln.strip()]
    else:
        sys.exit("Provide --text or --file")

    summarize_fn = summarize_with_llm if args.llm else summarize
    out = []
    for r in clf.predict(messages):
        s = summarize_fn(r["log_message"], r["predicted_label"], r["confidence"])
        s["needs_review"] = r["needs_review"]
        out.append(s)
    print(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser(description="Log root-cause classification + summary pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="train, cross-validate, write metrics").set_defaults(func=cmd_train)
    sub.add_parser("evaluate", help="print saved metrics").set_defaults(func=cmd_evaluate)

    pp = sub.add_parser("predict", help="classify + summarize log(s)")
    pp.add_argument("--text", help="a single log message")
    pp.add_argument("--file", help="path to a file with one log per line")
    pp.add_argument("--llm", action="store_true", help="use local Ollama for the summary")
    pp.set_defaults(func=cmd_predict)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
