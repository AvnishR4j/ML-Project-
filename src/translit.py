"""Devanagari -> informal Roman ("Hinglish") transliteration, so Hindi-script messages can use the Hinglish models.

Two parts:
  1. DICT   the most frequent Hindi words, written the way real Hinglish writers spell them. The spelling that wins in
            L3Cube-HingLID Hindi-tagged text was chosen (for example 'nahi' 8,852 vs 'nahin' 658; 'kuch' 3,920 vs 'kuchh' 435).
  2. rules  letter-by-letter mapping with schwa deletion (kamal, not kamala) for every other word.
Latin text, digits and punctuation pass through unchanged (mixed-script messages keep their English words).
The JavaScript copy used in the browser (site/engine.js) is generated from these tables by scripts/gen_translit_js.py,
and scripts/web_parity.py checks that both give identical output.
"""
import re, unicodedata

CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
        "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
        "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
        "ष": "sh", "स": "s", "ह": "h", "ळ": "l"}
NUKTA = {"क": "k", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"}      # consonant + U+093C
VOWEL = {"अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}
MATRA = {"ा": "aa", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e"}
DIGITS = {chr(0x966 + i): str(i) for i in range(10)}
HALANT, NUKTA_CH, ANUSVARA, CHANDRABINDU, VISARGA = "्", "़", "ं", "ँ", "ः"
LABIAL = set("पफबभम")

DICT = {
    # function words and pronouns
    "है": "hai", "हैं": "hain", "था": "tha", "थी": "thi", "थे": "the", "हो": "ho", "हूँ": "hu", "हुआ": "hua", "हुई": "hui", "होगा": "hoga", "होगी": "hogi",
    "नहीं": "nahi", "ना": "na", "न": "na", "भी": "bhi", "ही": "hi", "तो": "to", "और": "aur", "या": "ya", "कि": "ki", "की": "ki", "का": "ka", "के": "ke",
    "को": "ko", "से": "se", "में": "me", "पर": "par", "पे": "pe", "ने": "ne", "तक": "tak", "यह": "yeh", "ये": "ye", "वह": "wo", "वो": "wo", "वे": "wo",
    "इस": "is", "उस": "us", "यहाँ": "yahan", "वहाँ": "wahan", "कहाँ": "kahan", "कब": "kab", "कैसे": "kaise", "कैसा": "kaisa", "कैसी": "kaisi", "क्या": "kya",
    "क्यों": "kyu", "क्योंकि": "kyunki", "कौन": "kaun", "कितना": "kitna", "कितने": "kitne", "कुछ": "kuch", "सब": "sab", "सबको": "sabko", "मैं": "main",
    "मुझे": "mujhe", "मुझको": "mujhko", "मेरा": "mera", "मेरी": "meri", "मेरे": "mere", "हम": "hum", "हमारा": "hamara", "हमारी": "hamari", "हमें": "hume",
    "तुम": "tum", "तुम्हें": "tumhe", "तुम्हारा": "tumhara", "तुम्हारी": "tumhari", "आप": "aap", "आपका": "aapka", "आपकी": "aapki", "आपके": "aapke",
    "आपको": "aapko", "अपना": "apna", "अपनी": "apni", "अपने": "apne", "उसका": "uska", "उसकी": "uski", "इसका": "iska", "इसकी": "iski", "उन्हें": "unhe",
    "लिए": "liye", "लिये": "liye", "बहुत": "bahut", "बहोत": "bahut", "बात": "baat", "काम": "kaam", "साथ": "sath", "आज": "aaj", "कल": "kal",
    "अब": "ab", "अभी": "abhi", "जब": "jab", "तब": "tab", "लेकिन": "lekin", "मगर": "magar", "अगर": "agar", "फिर": "phir", "बस": "bas",
    "बिल्कुल": "bilkul", "बिलकुल": "bilkul", "सिर्फ": "sirf", "सिर्फ़": "sirf", "जरूर": "zaroor", "ज़रूर": "zaroor", "शुक्रिया": "shukriya",
    "धन्यवाद": "dhanyavad", "नमस्ते": "namaste", "दोस्त": "dost", "भाई": "bhai", "पापा": "papa", "मम्मी": "mummy", "माँ": "maa", "घर": "ghar",
    "पैसा": "paisa", "पैसे": "paise", "ठीक": "thik", "खाना": "khana", "शादी": "shadi", "ज्यादा": "jyada", "ज़्यादा": "jyada", "पहले": "pehle",
    "रहा": "raha", "रहे": "rahe", "रही": "rahi", "गया": "gaya", "गई": "gayi", "गये": "gaye", "गए": "gaye", "करो": "karo", "करना": "karna", "करें": "kare",
    "करे": "kare", "कर": "kar", "दो": "do", "एक": "ek", "दिन": "din", "रात": "raat", "समय": "samay", "लोग": "log", "सा": "sa", "जी": "ji",
    "जा": "ja", "जाओ": "jao", "आओ": "aao", "आया": "aaya", "आई": "aayi", "आए": "aaye", "चाहिए": "chahiye", "सकता": "sakta", "सकती": "sakti",
    "वाला": "wala", "वाली": "wali", "वाले": "wale", "ऐसा": "aisa", "ऐसे": "aise", "जैसे": "jaise", "इतना": "itna", "बड़ा": "bada", "बड़ी": "badi",
    # opinion words
    "अच्छा": "acha", "अच्छी": "achi", "अच्छे": "ache", "बुरा": "bura", "बुरी": "buri", "खराब": "kharab", "ख़राब": "kharab", "बेकार": "bekar",
    "बकवास": "bakwas", "मस्त": "mast", "शानदार": "shandar", "ज़बरदस्त": "zabardast", "जबरदस्त": "zabardast", "प्यार": "pyar", "खुश": "khush",
    "दुख": "dukh", "सुंदर": "sundar", "बढ़िया": "badhiya", "कमाल": "kamaal", "गंदा": "ganda", "धोखा": "dhokha", "परेशान": "pareshan", "मजा": "maza", "मज़ा": "maza",
    # loan words written in Devanagari (spelled the English way)
    "फ़ोन": "phone", "फोन": "phone", "मोबाइल": "mobile", "टाइम": "time", "स्कूल": "school", "ऑफिस": "office", "बैंक": "bank", "अकाउंट": "account",
    "पासवर्ड": "password", "लॉटरी": "lottery", "कंपनी": "company", "ऑफर": "offer", "मैसेज": "message", "लिंक": "link", "क्लिक": "click", "इंस्टॉल": "install",
    "ऐप": "app", "डिलीवरी": "delivery", "ऑर्डर": "order", "पैकेज": "package", "रिफंड": "refund", "कैशबैक": "cashback", "पेमेंट": "payment", "ओटीपी": "otp",
    "केवाईसी": "kyc", "लोन": "loan", "बैलेंस": "balance", "बिल": "bill", "रिचार्ज": "recharge", "सिम": "sim", "इनाम": "inaam", "फ्री": "free", "कार्ड": "card",
    "नंबर": "number", "वीडियो": "video", "फिल्म": "film", "फ़िल्म": "film", "गाना": "gana", "मैच": "match", "टीम": "team", "क्रिकेट": "cricket", "ट्रेन": "train",
    "टिकट": "ticket", "डॉक्टर": "doctor", "हॉस्पिटल": "hospital", "दवा": "dawa", "सर": "sir", "मैडम": "madam", "प्लीज": "please", "सॉरी": "sorry", "थैंक्स": "thanks",
}

_DEV = re.compile(r"[ऀ-ॿ]+")


def _units(word):
    """split a Devanagari word into units: {'c': consonant roman or None, 'v': explicit vowel, 'h': halant, 'n': nasal, 'x': visarga}"""
    U, i, w = [], 0, word
    while i < len(w):
        ch = w[i]
        if ch in CONS:
            base, i = CONS[ch], i + 1
            if i < len(w) and w[i] == NUKTA_CH:
                base = NUKTA.get(ch, base); i += 1
            U.append({"c": base, "d": ch, "v": None, "h": False, "n": "", "x": ""})
        elif ch in VOWEL:
            U.append({"c": None, "d": "", "v": VOWEL[ch], "h": False, "n": "", "x": ""}); i += 1
        elif ch in MATRA and U:
            U[-1]["v"] = MATRA[ch]; i += 1
        elif ch == HALANT and U:
            U[-1]["h"] = True; i += 1
        elif ch in (ANUSVARA, CHANDRABINDU) and U:
            U[-1]["n"] = "a" if ch == ANUSVARA else "c"; i += 1
        elif ch == VISARGA and U:
            U[-1]["x"] = "h"; i += 1
        elif ch == NUKTA_CH or ch in "‌‍":
            i += 1
        else:
            i += 1
    return U


def _word(word):
    word = unicodedata.normalize("NFC", word)
    if word in DICT:
        return DICT[word]
    U = _units(word); n = len(U)
    cand = [u["c"] is not None and u["v"] is None and not u["h"] for u in U]      # consonant with an inherent 'a'
    keep = list(cand)                                                             # does the inherent 'a' survive?
    has_v = lambda j: U[j]["v"] is not None or keep[j]
    for j in range(n - 1, -1, -1):
        if not cand[j]:
            continue
        if j == n - 1:
            keep[j] = n == 1                                                           # final schwa is dropped (kamal)
        elif j > 0 and has_v(j - 1) and has_v(j + 1) and not U[j + 1]["h"]:
            keep[j] = False                                                            # medial schwa between two vowels (samajhna)
    out = []
    for j, u in enumerate(U):
        if u["c"] is None:
            out.append(u["v"])
        else:
            c = u["c"]
            vow = u["v"] if u["v"] is not None else ("a" if keep[j] else "")
            if c == "v":
                c = "w" if vow in ("a", "aa", "o", "au") else "v"
            out.append(c + vow)
        if u["n"]:                                                                     # anusvara: m before a labial (sampark), else n (hindi)
            out.append("m" if (u["n"] == "a" and j + 1 < n and U[j + 1]["d"] in LABIAL) else "n")
        if u["x"]:
            out.append("h")
    s = "".join(out).replace("chchh", "chh")
    return s[:-2] + "a" if len(s) > 3 and s.endswith("aa") else s                    # samajhnaa -> samajhna (final long a is usually written short)


def deva2roman(text):
    """transliterate every Devanagari run in text; everything else is kept as is"""
    t = "".join(DIGITS.get(c, c) for c in text).replace("।", ".").replace("॥", ".")
    return _DEV.sub(lambda m: _word(m.group(0)), t)


def has_devanagari(text):
    return bool(_DEV.search(text))


if __name__ == "__main__":
    for s in ["कमल", "समझना", "नमस्ते दोस्तों", "यह फ़ोन बहुत अच्छा है", "खराब सर्विस बकवास", "सुंदर", "हिंदी में बात करो", "आपका बैंक अकाउंट बंद हो जाएगा", "पैसे भेजो ₹500 अभी", "Bhai, शादी की shopping कब करने जाएं?"]:
        print(s, "->", deva2roman(s))
