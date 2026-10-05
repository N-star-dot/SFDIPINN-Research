"""
Build the fast per-cell .npz cache (sfdi/ingest.py) from the lab's raw
hyperstacks. Run this once after pointing config.yaml at the real data, and
again whenever the raw files change -- it's incremental, so an unchanged cell
is skipped.

    python scripts/ingest_data.py            # only changed/new cells
    python scripts/ingest_data.py --force    # re-ingest every cell
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sfdi.common import load_config
from sfdi import ingest

p = argparse.ArgumentParser()
p.add_argument("--force", action="store_true", help="ignore the existing manifest, re-ingest every cell")
p.add_argument("--config", default=None, help="path to a config.yaml (default: $SFDI_CONFIG or config.yaml)")
a = p.parse_args()

cfg = load_config(a.config)
t0 = time.time()
man = ingest.run(cfg, force=a.force)
elapsed = time.time() - t0

print(f"{'cell':<18}{'ROI px':>10}{'% capped @471nm':>17}  warnings")
for name, c in sorted(man["cells"].items()):
    pct = 100 * c["ref_capped_pixels"][0] / c["roi_pixels"] if c.get("ref_capped_pixels") and c["roi_pixels"] else 0.0
    warn = "; ".join(c.get("warnings", [])) or "-"
    print(f"{name:<18}{c['roi_pixels']:>10}{pct:>16.1f}%  {warn}")

if man["skipped"]:
    print("\nskipped:")
    for s in man["skipped"]:
        print(f"  {s['path']}: {s['reason']}")

print(f"\ningested {man['stats']['ingested']}, up to date {man['stats']['up_to_date']}, "
      f"{len(man['cells'])} cells total, {elapsed:.1f}s "
      f"-> {os.path.relpath(cfg['processed_root'])}")
