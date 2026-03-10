"""
predict_cli.py
--------------
Command-line tool to check URLs without running the web server.

Usage:
    python predict_cli.py https://www.google.com
    python predict_cli.py http://secure-paypal-login.xyz/verify
    python predict_cli.py --batch urls.txt
"""

import sys
import os
import pickle
import argparse
import numpy as np
from feature_extractor import features_to_vector, extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("ERROR: model.pkl not found. Run: python train_model.py")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_url(model, url: str, verbose: bool = True) -> dict:
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    X = np.array([features_to_vector(url)])
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    phish_prob = float(proba[1])

    result = {
        "url": url,
        "prediction": "PHISHING" if pred == 1 else "LEGITIMATE",
        "phishing_probability": round(phish_prob * 100, 1),
    }

    if verbose:
        icon = "🚨" if pred == 1 else "✅"
        color = "\033[91m" if pred == 1 else "\033[92m"
        reset = "\033[0m"
        print(f"\n{icon}  {color}{result['prediction']}{reset}")
        print(f"   URL:         {url}")
        print(f"   Phishing %:  {result['phishing_probability']}%")

        # Show top features
        feats = extract_features(url)
        print("\n   Key Features:")
        for k, v in list(feats.items())[:8]:
            print(f"     {k:<30} = {v}")

    return result


def batch_predict(model, filepath: str):
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    with open(filepath) as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"\nProcessing {len(urls)} URLs...\n")
    print(f"{'URL':<60} {'RESULT':<12} {'PHISH %':>8}")
    print("-" * 84)

    for url in urls:
        r = predict_url(model, url, verbose=False)
        color = "\033[91m" if r["prediction"] == "PHISHING" else "\033[92m"
        reset = "\033[0m"
        print(f"{r['url']:<60} {color}{r['prediction']:<12}{reset} {r['phishing_probability']:>7}%")


def main():
    parser = argparse.ArgumentParser(
        description="Phishing URL Detector - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_cli.py https://www.google.com
  python predict_cli.py http://secure-paypal-login.xyz/verify
  python predict_cli.py --batch urls.txt
        """
    )
    parser.add_argument("url", nargs="?", help="URL to analyze")
    parser.add_argument("--batch", metavar="FILE", help="Text file with one URL per line")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    if not args.url and not args.batch:
        parser.print_help()
        sys.exit(0)

    model_data = load_model()
    model = model_data["model"]

    if args.batch:
        batch_predict(model, args.batch)
    elif args.url:
        predict_url(model, args.url, verbose=not args.quiet)


if __name__ == "__main__":
    main()