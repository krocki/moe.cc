#!/usr/bin/env python3
# merge_dir.py — merge per-tensor dumps (export.py --outdir) into one .bin

import sys, os, json, struct
try:
  from tqdm import tqdm  # progress bar
  _HAS_TQDM = True
except Exception:
  _HAS_TQDM = False

MAGIC = b"QW3W\x00\x02"

def read_one_tensor(path: str) -> bytes:
  with open(path, "rb") as f:
    if f.read(6) != MAGIC:
      raise ValueError(f"{path}: bad magic")
    (nt,) = struct.unpack("<I", f.read(4))
    if nt != 1:
      raise ValueError(f"{path}: expected exactly 1 tensor, got {nt}")
    return f.read()  # rest of single-tensor blob

def main():
  if len(sys.argv) != 3:
    print("Usage: merge_dir.py <out.bin> <dump_dir>")
    sys.exit(1)
  out_path, dump_dir = sys.argv[1], sys.argv[2]
  man_path = os.path.join(dump_dir, "manifest.json")
  with open(man_path, "r") as f:
    manifest = json.load(f)

  tensors = manifest["tensors"]  # list of {name,file,...}
  n = len(tensors)

  with open(out_path, "wb") as out:
    out.write(MAGIC)
    # FIX: correct struct format (little-endian unsigned int)
    out.write(struct.pack("<I", n))

    iterator = tqdm(tensors, desc="Merging", unit="tensor") if _HAS_TQDM else tensors
    for t in iterator:
      path = os.path.join(dump_dir, t["file"])
      blob = read_one_tensor(path)
      out.write(blob)

  print(f"Wrote {out_path} with {n} tensors (from {dump_dir})")

if __name__ == "__main__":
  main()
