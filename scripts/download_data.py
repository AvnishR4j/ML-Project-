"""Download + unpack every dataset into data/raw (works on Windows/Mac/Linux).   python scripts/download_data.py

Sources (all public; check each licence before redistributing):
  UCI SMS Spam Collection        via github.com/justmarkham/pycon-2016-tutorial
  NLTK corpora (UDHR, movie_reviews, twitter_samples, sentence_polarity)  via github.com/nltk/nltk_data
  L3Cube-HingLID                 github.com/l3cube-pune/code-mixed-nlp   (CC BY-NC-SA 4.0; cite Nayak & Joshi, 2022)
  IIIT-H code-mixed sentiment    github.com/DrImpossible/Sub-word-LSTM   (Prabhu et al., 2016)
  TweetEval sentiment (SemEval-2017) github.com/cardiffnlp/tweeteval  (English tweets, larger training set; Rosenthal et al., 2017)
  Hinglish / Hindi scam lines    github.com/gupta7050/multilingual-phishing-detection  (small hand-written challenge sets)
"""
import urllib.request, zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
GH = "https://raw.githubusercontent.com/"
NLTK = GH + "nltk/nltk_data/gh-pages/packages/corpora/{}.zip"
HING = GH + "l3cube-pune/code-mixed-nlp/main/L3Cube-HingLID/{}.txt"
FILES = {
    "sms.tsv": GH + "justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    "udhr.zip": NLTK.format("udhr"),
    "movie_reviews.zip": NLTK.format("movie_reviews"),
    "twitter_samples.zip": NLTK.format("twitter_samples"),
    "sentence_polarity.zip": NLTK.format("sentence_polarity"),
    "hinglid/train.txt": HING.format("train"),
    "hinglid/validation.txt": HING.format("validation"),
    "hinglid/test.txt": HING.format("test"),
    "iiith_codemixed.txt": GH + "DrImpossible/Sub-word-LSTM/master/Data/IIITH_Codemixed.txt",
    **{f"tweeteval/{sp}_{k}.txt": GH + f"cardiffnlp/tweeteval/main/datasets/sentiment/{sp}_{k}.txt" for sp in ("train", "val", "test") for k in ("text", "labels")},
    "hinglish_scam_challenge_raw.csv": GH + "gupta7050/multilingual-phishing-detection/main/hinglish_data.csv",
    "hindi_scam_challenge_raw.csv": GH + "gupta7050/multilingual-phishing-detection/main/hindi_data.csv",
}
for name, url in FILES.items():
    dest = RAW / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print("downloading", name)
        urllib.request.urlretrieve(url, dest)
    if name.endswith(".zip"):
        zipfile.ZipFile(dest).extractall(RAW)
print("done ->", RAW)
