#!/usr/bin/env python3
# merge_bins.py — merge multiple export.py .bin files into one

import sys, struct

MAGIC = b"QW3W\x00\x01"

def read_exact(f, n):
  b = f.read(n)
  if len(b) != n:
    raise EOFError("unexpected EOF")
  return b

def load_bin(path):
  with open(path, "rb") as f:
    magic = read_exact(f, 6)
    if magic != MAGIC:
      raise ValueError(f"{path}: bad magic")
    (nt,) = struct.unpack("<I", read_exact(f, 4))
    tensors = []
    for _ in range(nt):
      (name_len,) = struct.unpack("<I", read_exact(f, 4))
      name = read_exact(f, name_len).decode("utf-8")
      dt, nd = struct.unpack("<II", read_exact(f, 8))
      shape = struct.unpack("<" + "I"*nd, read_exact(f, 4*nd))
      # compute nbytes from dtype
      if dt == 0: bpe = 4   # f32
      elif dt == 1: bpe = 2 # f16
      elif dt == 2: bpe = 1 # i8
      elif dt == 3: bpe = 1 # i4 (packed)
      else: raise ValueError(f"{path}: unknown dtype {dt}")
      elems = 1
      for s in shape: elems *= s
      nbytes = elems * bpe
      data = read_exact(f, nbytes)
      tensors.append((name, dt, nd, shape, data))
    return tensors

def main():
  if len(sys.argv) < 3:
    print("Usage: merge_bins.py <out.bin> <in1.bin> [in2.bin ...]")
    sys.exit(1)
  out = sys.argv[1]
  ins = sys.argv[2:]

  merged = {}
  order = []
  for p in ins:
    for (name, dt, nd, shape, data) in load_bin(p):
      if name in merged:
        raise SystemExit(f"duplicate tensor name across inputs: {name}")
      merged[name] = (dt, nd, shape, data)
      order.append(name)

  with open(out, "wb") as f:
    f.write(MAGIC)
    f.write(struct.pack("<I", len(order)))
    for name in order:
      dt, nd, shape, data = merged[name]
      nb = name.encode("utf-8")
      f.write(struct.pack("<I", len(nb))); f.write(nb)
      f.write(struct.pack("<I", dt))
      f.write(struct.pack("<I", nd))
      f.write(struct.pack("<" + "I"*nd, *shape))
      f.write(data)
  print(f"Wrote {out} with {len(order)} tensors from {len(ins)} files")

if __name__ == "__main__":
  main()
