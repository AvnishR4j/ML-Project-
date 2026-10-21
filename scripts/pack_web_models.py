"""Pack web/models/*.json into gzip+base64 wrappers (web/packed/*.json) that the page unpacks with DecompressionStream."""
import base64, gzip, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; out = ROOT / "web" / "packed"; out.mkdir(exist_ok=True)
tot = 0
for p in sorted((ROOT / "web" / "models").glob("*.json")):
    z = base64.b64encode(gzip.compress(p.read_bytes(), 9)).decode()
    (out / p.name).write_text(json.dumps({"gz": z})); tot += len(z); print(p.name, round(len(z) / 1e6, 2), "MB")
print("total", round(tot / 1e6, 2), "MB")
