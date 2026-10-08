import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import make_work_plant_runs as W
rows = W.check([W.SKIN_DIR])
out = ROOT / "docs/experiments/20261008-contact_model_policies/arm_scene_checks.json"
prev = json.loads(out.read_text())["checks"] if out.exists() else []
prev = [r for r in prev if r["run"] != rows[0]["run"]] + rows
out.write_text(json.dumps({"checks": prev}, indent=1))
