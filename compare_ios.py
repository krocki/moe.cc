#!/usr/bin/env python3
import argparse, numpy as np

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--x1", required=True)
  ap.add_argument("--y1", required=True)
  ap.add_argument("--x2", required=True)
  ap.add_argument("--y2", required=True)
  ap.add_argument("--tol", type=float, default=1e-6)
  args = ap.parse_args()

  x1 = np.load(args.x1); y1 = np.load(args.y1)
  x2 = np.load(args.x2); y2 = np.load(args.y2)

  if x1.shape != x2.shape:
    print(f"x shape mismatch: {x1.shape} vs {x2.shape}"); return
  if y1.shape != y2.shape:
    print(f"y shape mismatch: {y1.shape} vs {y2.shape}"); return

  xdiff = np.max(np.abs(x1.astype(np.float64) - x2.astype(np.float64)))
  ydiff = np.max(np.abs(y1.astype(np.float64) - y2.astype(np.float64)))

  print(f"x max abs diff: {xdiff:.6g}")
  print(f"y max abs diff: {ydiff:.6g}")
  print("PASS" if (xdiff <= args.tol and ydiff <= args.tol) else "FAIL")

if __name__ == "__main__":
  main()
