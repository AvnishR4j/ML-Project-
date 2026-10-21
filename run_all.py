"""Reproduce everything:  python run_all.py          (about 15-20 minutes on a laptop CPU, no GPU needed)

Order matters: data -> baselines -> experiments -> deployed models -> end-to-end evaluation -> README.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("download data", ["scripts/download_data.py"]),
    ("language + spam baselines (NB / SVM / KNN)", ["src/train.py"]),
    ("sentiment experiments (Hinglish + English cross-domain)", ["src/exp_sentiment.py"]),
    ("spam transfer experiments", ["src/exp_spam.py"]),
    ("theme-discovery experiment", ["src/exp_cluster.py"]),
    ("deployed language model (reject option)", ["src/final_language.py"]),
    ("deployed spam model (calibrated, per-language thresholds)", ["src/final_spam.py"]),
    ("deployed theme models", ["src/final_themes.py"]),
    ("end-to-end evaluation", ["src/final_e2e.py"]),
    ("README from results", ["scripts/make_readme.py"]),
]
for name, cmd in STEPS:
    print(f"\n=== {name} ===", flush=True)
    cwd = ROOT / "src" if cmd[0].startswith("src/") else ROOT
    script = ROOT / cmd[0]
    subprocess.run([sys.executable, str(script)], cwd=cwd, check=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
print("\nAll done. Start the app:  streamlit run app/streamlit_app.py")
