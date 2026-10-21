"""Download + unpack the prototype datasets into data/raw (works on Windows/Mac/Linux).
   python scripts/download_data.py
"""
import urllib.request, zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
NLTK = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/{}.zip"
FILES = {
    "sms.tsv": "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    "udhr.zip": NLTK.format("udhr"),
    "movie_reviews.zip": NLTK.format("movie_reviews"),
}
for name, url in FILES.items():
    dest = RAW / name
    if not dest.exists():
        print("downloading", name)
        urllib.request.urlretrieve(url, dest)
    if name.endswith(".zip"):
        zipfile.ZipFile(dest).extractall(RAW)
print("done ->", RAW)
