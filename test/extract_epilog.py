#!/usr/bin/env python3
import re
inpath = "debug_epipolar_lines.txt"
out_edges = "ep_edges.txt"
out_lines = "ep_lines.txt"

pat = re.compile(r'.*at\s*\(\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\)\s*:\s*(.*)')
edges = []
lines = []
with open(inpath, 'r') as f:
    for ln in f:
        m = pat.match(ln.strip())
        if not m: 
            continue
        x,y,rest = m.groups()
        nums = rest.strip().split()
        if len(nums) < 3:
            continue
        a,b,c = nums[:3]
        edges.append((float(x), float(y)))
        lines.append((float(a), float(b), float(c)))

with open(out_edges, 'w') as f:
    for x,y in edges:
        f.write(f"{x} {y}\n")
with open(out_lines, 'w') as f:
    for a,b,c in lines:
        f.write(f"{a} {b} {c}\n")

print(f"Wrote {len(edges)} edges -> {out_edges} and {len(lines)} lines -> {out_lines}")