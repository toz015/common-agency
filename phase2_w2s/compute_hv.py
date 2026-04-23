"""
Compute 2D hypervolume for PARM vs GenARM on the (help, safety) Pareto.
safety = -harm, so both axes are "higher = better". Reference point is
(min_help, min_safety) across both methods, shifted by a small margin so HV
is strictly positive.

Reports HV for: raw means, drift-filtered means. Both methods use the same
reference point so HV numbers are directly comparable.
"""
import csv
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "phase2_results_n100" / "results_phase2_n100_t512" / "pareto_n100_filtered.csv"


def hv_2d(points, ref):
    """2D hypervolume, both axes maximize. Reference = (rx, ry).
    HV = area of union of rectangles [rx, xi] x [ry, yi] for non-dominated pts.
    Formula: sum (x_i - x_{i-1}) * (y_i - ry) after sorting x asc (y desc),
    with x_0 = rx."""
    # keep only points strictly dominating ref
    pts = [(x, y) for (x, y) in points if x > ref[0] and y > ref[1]]
    if not pts:
        return 0.0
    # non-dominated filter: walk right-to-left, keep points with strictly
    # larger y than any seen so far (to their right).
    pts.sort(key=lambda p: p[0])
    nd = []
    max_y_seen = float("-inf")
    for x, y in reversed(pts):
        if y > max_y_seen:
            nd.append((x, y))
            max_y_seen = y
    nd.reverse()  # now x ascending, y descending
    # accumulate
    hv = 0.0
    prev_x = ref[0]
    for x, y in nd:
        hv += (x - prev_x) * (y - ref[1])
        prev_x = x
    return hv


rows = list(csv.DictReader(open(CSV)))
parm_raw   = [(float(r["help_all"]),   float(r["safety_all"]))   for r in rows if r["method"] == "PARM"]
parm_clean = [(float(r["help_clean"]), float(r["safety_clean"])) for r in rows if r["method"] == "PARM"]
gen_raw    = [(float(r["help_all"]),   float(r["safety_all"]))   for r in rows if r["method"] == "GenARM"]
gen_clean  = [(float(r["help_clean"]), float(r["safety_clean"])) for r in rows if r["method"] == "GenARM"]

# shared reference: min over both methods' both axes, minus margin
all_pts = parm_raw + parm_clean + gen_raw + gen_clean
ref_help  = min(p[0] for p in all_pts) - 0.5
ref_safe  = min(p[1] for p in all_pts) - 0.5
REF = (ref_help, ref_safe)

print(f"reference point (help, safety) = ({ref_help:.3f}, {ref_safe:.3f})\n")

print(f"{'metric':<30} {'PARM':>10} {'GenARM':>10} {'Δ (G-P)':>10}")
print("-" * 65)
for label, parm_pts, gen_pts in [
    ("HV (raw means)",          parm_raw,   gen_raw),
    ("HV (drift-filtered)",     parm_clean, gen_clean),
]:
    hp = hv_2d(parm_pts, REF)
    hg = hv_2d(gen_pts, REF)
    print(f"{label:<30} {hp:>10.3f} {hg:>10.3f} {hg - hp:>+10.3f}")

# Also show the Pareto points for reference
print("\nPareto points (help, safety):")
for name, pts in [("PARM raw", parm_raw), ("PARM clean", parm_clean),
                   ("GenARM raw", gen_raw), ("GenARM clean", gen_clean)]:
    print(f"  {name:<12} " + "  ".join(f"({x:+.2f}, {y:+.2f})" for x, y in pts))
