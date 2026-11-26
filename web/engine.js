/* In-browser port of the Python pipeline (src/pipeline.py): language -> spam -> sentiment -> theme.
 * Weights come from scripts/export_web_models.py. Parity with Python is checked by scripts/web_parity.py.
 * Works in a browser (global MIEngine) and in Node (module.exports). */
(function (root) {
  const WORD = /[\p{L}\p{N}_]{2,}/gu;
  const W = "[\\p{L}\\p{N}_]";           // Python \w (Unicode)
  const D = "\\p{Nd}";
  const REV = { "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "і": "i" };

  function bytes(b64) {
    if (typeof Buffer !== "undefined") return new Uint8Array(Buffer.from(b64, "base64"));
    const s = atob(b64), u = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
    return u;
  }
  const f32 = (b64) => { const u = bytes(b64); return new Float32Array(u.buffer, u.byteOffset, u.byteLength >> 2); };
  const i8 = (b64) => { const u = bytes(b64); return new Int8Array(u.buffer, u.byteOffset, u.byteLength); };
  const i16 = (b64) => { const u = bytes(b64); return new Int16Array(u.buffer, u.byteOffset, u.byteLength >> 1); };
  const cp = (s) => Array.from(s);

  // ---------------------------------------------------------------- text helpers (src/text_norm.py)
  function fixSpacing(text) {
    const t = text.normalize("NFKC");
    const segs = t.split(/ {2,}/);
    const out = segs.map((seg) => {
      const toks = seg.split(" ");
      if (toks.length >= 3 && toks.every((x) => cp(x).length === 1)) return toks.join("");
      return seg;
    });
    return out.join(" ");
  }
  function normalize(text) {
    let t = text.normalize("NFKC");
    t = cp(t).map((c) => REV[c] || c).join("");
    return fixSpacing(t);
  }
  const squeeze = (t) => t.replace(/(.)\1{2,}/gu, "$1$1");
  const prepFns = { lower: (t) => t.toLowerCase(), prep: (t) => squeeze(t.toLowerCase()) };

  // ---------------------------------------------------------------- TF-IDF vectoriser (sklearn semantics)
  // negation-aware preprocessing (mirror of src/negation.py NegPrep)
  const EN_NEG = new Set("not no never nothing neither nor without cannot hardly barely dont doesnt didnt isnt wasnt arent werent cant wont wouldnt shouldnt couldnt aint nobody".split(" "));
  const STOP = new Set("the a an is was are were be been am it this that to at all very so too really much any i do does did have has had will would can could should hai tha thi ho hain hota hoti bahut bohot ek ye yeh ki ka ke".split(" "));
  const CONTR = [[/\bcan'?t\b/g, "can not"], [/\bwon'?t\b/g, "will not"], [/\bain'?t\b/g, "is not"], [/n['\u2019]t\b/g, " not"], [/\bcannot\b/g, "can not"]];
  const NTOK = /[\p{L}\p{N}]+|[^\p{L}\p{N}_\s]/gu, NWORD = /^[\p{L}\p{N}_]/u;
  function negPrep(pol, scope) {
    const P = new Map(Object.entries(pol));
    return (text) => {
      let t = squeeze(text.toLowerCase()); for (const [a, b] of CONTR) t = t.replace(a, b);
      const out = []; let left = 0;
      for (const w of t.match(NTOK) || []) {
        if (!NWORD.test(w)) { left = 0; out.push(w); continue; }
        if (EN_NEG.has(w)) { left = scope; out.push(w); continue; }
        if (left > 0) {
          if (STOP.has(w)) { out.push(w); continue; }
          out.push("not_" + w); const p = P.get(w) || 0; if (p) out.push(p > 0 ? "negflip_pos" : "negflip_neg"); left--;
        } else out.push(w);
      }
      return out.join(" ");
    };
  }
  function loadVec(v) {
    return { an: v.an, ng: v.ng, prep: v.prep === "neg" ? negPrep(v.pol, v.scope) : prepFns[v.prep], idx: new Map(v.vocab.map((t, i) => [t, i])), idf: f32(v.idf), n: v.vocab.length };
  }
  function grams(v, text) {
    const t = v.prep(text), out = [];
    if (v.an === "word") {
      const tokens = t.match(WORD) || [];
      for (let n = v.ng[0]; n <= v.ng[1]; n++) {
        if (n === 1) out.push(...tokens);
        else for (let i = 0; i + n <= tokens.length; i++) out.push(tokens.slice(i, i + n).join(" "));
      }
    } else {                                   // char_wb
      const words = t.replace(/\s\s+/g, " ").trim().split(/\s+/).filter(Boolean);
      for (const word of words) {
        const w = cp(" " + word + " "), len = w.length;
        for (let n = v.ng[0]; n <= v.ng[1]; n++) {
          let off = 0;
          out.push(w.slice(off, off + n).join(""));
          while (off + n < len) { off++; out.push(w.slice(off, off + n).join("")); }
          if (off === 0) break;
        }
      }
    }
    return out;
  }
  function vectorize(v, text) {                 // -> {idx: Int32Array, val: Float64Array} (L2-normalised, sublinear tf)
    const counts = new Map();
    for (const g of grams(v, text)) { const i = v.idx.get(g); if (i !== undefined) counts.set(i, (counts.get(i) || 0) + 1); }
    const idx = [], val = []; let s = 0;
    for (const [i, c] of counts) { const x = (1 + Math.log(c)) * v.idf[i]; idx.push(i); val.push(x); s += x * x; }
    const nrm = Math.sqrt(s) || 1;
    return { idx, val: val.map((x) => x / nrm) };
  }

  // ---------------------------------------------------------------- structural spam cues (src/spam_features.py)
  const R = {
    url: new RegExp(`https?://|www\\.|(?<!${W})${W}+\\.(?:com|in|co|net|org|link|xyz|top|info)(?!${W})|(?<!${W})bit\\.ly|(?<!${W})tinyurl`, "iu"),
    phone: new RegExp(`(?<!${D})(?:\\+?91[\\s-]?)?[6-9]${D}{9}(?!${D})|(?<!${W})${D}{5}[\\s-]?${D}{5}(?!${W})|(?<!${W})${D}{4,}(?!${W})`, "u"),
    money: new RegExp(`[₹$£€]|(?<!${W})rs\\.?\\s?${D}|(?<!${W})inr(?!${W})|(?<!${W})lakh|(?<!${W})crore|(?<!${W})paise|रु`, "iu"),
    pct: new RegExp(`${D}+\\s?%`, "u"),
    lex: new RegExp(
      `(?<!${W})(?:inaam|inam|jeeto|jeet\\s?gaye|jeete|muft|free|lottery|lucky|winner|kyc|otp|pan\\s?card|aadhaar|aadhar|` +
      `band\\s?ho\\s?jayega|block\\s?ho|turant|jaldi|abhi\\s?click|click\\s?karein|click\\s?kare|link\\s?par|` +
      `claim|offer|subsidy|yojana|loan|cashback|refund|expire|verify|update\\s?karein|apply\\s?now|register\\s?kar|` +
      `guaranteed|return|invest|crore|lakh|prize)(?!${W})` +
      `|इनाम|लॉटरी|मुफ्त|तुरंत|क्लिक|लिंक|केवाईसी|बंद हो|रिटर्न|अभी`, "giu"),
  };
  const isAlpha = (c) => /^\p{L}$/u.test(c);
  const isUpper = (c) => c !== c.toLowerCase() && c === c.toUpperCase();
  function structural(t) {
    const chars = cp(t), n = Math.max(chars.length, 1);
    const letters = chars.filter(isAlpha);
    const up = letters.filter(isUpper).length / Math.max(letters.length, 1);
    const digits = chars.filter((c) => /^\p{Nd}$/u.test(c)).length;
    const bang = (t.match(/!/g) || []).length;
    const lexN = (t.match(R.lex) || []).length;
    return [R.url.test(t) ? 1 : 0, R.phone.test(t) ? 1 : 0, R.money.test(t) ? 1 : 0, R.pct.test(t) ? 1 : 0,
      digits / n, up, Math.min(bang, 5) / 5, Math.log1p(chars.length) / 6, Math.min(lexN, 4) / 4];
  }

  /*TRANSLIT-START (generated by scripts/gen_translit_js.py from src/translit.py - do not edit by hand)*/
  const TL = { CONS: {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ळ": "l"}, NUKTA: {"क": "k", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"}, VOWEL: {"अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}, MATRA: {"ा": "aa", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e"}, DIGITS: {"०": "0", "१": "1", "२": "2", "३": "3", "४": "4", "५": "5", "६": "6", "७": "7", "८": "8", "९": "9"}, LABIAL: ["प", "फ", "ब", "भ", "म"],
    DICT: {"है": "hai", "हैं": "hain", "था": "tha", "थी": "thi", "थे": "the", "हो": "ho", "हूँ": "hu", "हुआ": "hua", "हुई": "hui", "होगा": "hoga", "होगी": "hogi", "नहीं": "nahi", "ना": "na", "न": "na", "भी": "bhi", "ही": "hi", "तो": "to", "और": "aur", "या": "ya", "कि": "ki", "की": "ki", "का": "ka", "के": "ke", "को": "ko", "से": "se", "में": "me", "पर": "par", "पे": "pe", "ने": "ne", "तक": "tak", "यह": "yeh", "ये": "ye", "वह": "wo", "वो": "wo", "वे": "wo", "इस": "is", "उस": "us", "यहाँ": "yahan", "वहाँ": "wahan", "कहाँ": "kahan", "कब": "kab", "कैसे": "kaise", "कैसा": "kaisa", "कैसी": "kaisi", "क्या": "kya", "क्यों": "kyu", "क्योंकि": "kyunki", "कौन": "kaun", "कितना": "kitna", "कितने": "kitne", "कुछ": "kuch", "सब": "sab", "सबको": "sabko", "मैं": "main", "मुझे": "mujhe", "मुझको": "mujhko", "मेरा": "mera", "मेरी": "meri", "मेरे": "mere", "हम": "hum", "हमारा": "hamara", "हमारी": "hamari", "हमें": "hume", "तुम": "tum", "तुम्हें": "tumhe", "तुम्हारा": "tumhara", "तुम्हारी": "tumhari", "आप": "aap", "आपका": "aapka", "आपकी": "aapki", "आपके": "aapke", "आपको": "aapko", "अपना": "apna", "अपनी": "apni", "अपने": "apne", "उसका": "uska", "उसकी": "uski", "इसका": "iska", "इसकी": "iski", "उन्हें": "unhe", "लिए": "liye", "लिये": "liye", "बहुत": "bahut", "बहोत": "bahut", "बात": "baat", "काम": "kaam", "साथ": "sath", "आज": "aaj", "कल": "kal", "अब": "ab", "अभी": "abhi", "जब": "jab", "तब": "tab", "लेकिन": "lekin", "मगर": "magar", "अगर": "agar", "फिर": "phir", "बस": "bas", "बिल्कुल": "bilkul", "बिलकुल": "bilkul", "सिर्फ": "sirf", "सिर्फ़": "sirf", "जरूर": "zaroor", "ज़रूर": "zaroor", "शुक्रिया": "shukriya", "धन्यवाद": "dhanyavad", "नमस्ते": "namaste", "दोस्त": "dost", "भाई": "bhai", "पापा": "papa", "मम्मी": "mummy", "माँ": "maa", "घर": "ghar", "पैसा": "paisa", "पैसे": "paise", "ठीक": "thik", "खाना": "khana", "शादी": "shadi", "ज्यादा": "jyada", "ज़्यादा": "jyada", "पहले": "pehle", "रहा": "raha", "रहे": "rahe", "रही": "rahi", "गया": "gaya", "गई": "gayi", "गये": "gaye", "गए": "gaye", "करो": "karo", "करना": "karna", "करें": "kare", "करे": "kare", "कर": "kar", "दो": "do", "एक": "ek", "दिन": "din", "रात": "raat", "समय": "samay", "लोग": "log", "सा": "sa", "जी": "ji", "जा": "ja", "जाओ": "jao", "आओ": "aao", "आया": "aaya", "आई": "aayi", "आए": "aaye", "चाहिए": "chahiye", "सकता": "sakta", "सकती": "sakti", "वाला": "wala", "वाली": "wali", "वाले": "wale", "ऐसा": "aisa", "ऐसे": "aise", "जैसे": "jaise", "इतना": "itna", "बड़ा": "bada", "बड़ी": "badi", "अच्छा": "acha", "अच्छी": "achi", "अच्छे": "ache", "बुरा": "bura", "बुरी": "buri", "खराब": "kharab", "ख़राब": "kharab", "बेकार": "bekar", "बकवास": "bakwas", "मस्त": "mast", "शानदार": "shandar", "ज़बरदस्त": "zabardast", "जबरदस्त": "zabardast", "प्यार": "pyar", "खुश": "khush", "दुख": "dukh", "सुंदर": "sundar", "बढ़िया": "badhiya", "कमाल": "kamaal", "गंदा": "ganda", "धोखा": "dhokha", "परेशान": "pareshan", "मजा": "maza", "मज़ा": "maza", "फ़ोन": "phone", "फोन": "phone", "मोबाइल": "mobile", "टाइम": "time", "स्कूल": "school", "ऑफिस": "office", "बैंक": "bank", "अकाउंट": "account", "पासवर्ड": "password", "लॉटरी": "lottery", "कंपनी": "company", "ऑफर": "offer", "मैसेज": "message", "लिंक": "link", "क्लिक": "click", "इंस्टॉल": "install", "ऐप": "app", "डिलीवरी": "delivery", "ऑर्डर": "order", "पैकेज": "package", "रिफंड": "refund", "कैशबैक": "cashback", "पेमेंट": "payment", "ओटीपी": "otp", "केवाईसी": "kyc", "लोन": "loan", "बैलेंस": "balance", "बिल": "bill", "रिचार्ज": "recharge", "सिम": "sim", "इनाम": "inaam", "फ्री": "free", "कार्ड": "card", "नंबर": "number", "वीडियो": "video", "फिल्म": "film", "फ़िल्म": "film", "गाना": "gana", "मैच": "match", "टीम": "team", "क्रिकेट": "cricket", "ट्रेन": "train", "टिकट": "ticket", "डॉक्टर": "doctor", "हॉस्पिटल": "hospital", "दवा": "dawa", "सर": "sir", "मैडम": "madam", "प्लीज": "please", "सॉरी": "sorry", "थैंक्स": "thanks"} };
  /*TRANSLIT-END*/

  // Devanagari -> informal Roman (mirror of src/translit.py)
  const TL_RUN = /[ऀ-ॿ]+/g;
  function tlUnits(w) {
    const U = []; const cs = Array.from(w); let i = 0;
    while (i < cs.length) {
      const ch = cs[i];
      if (TL.CONS[ch] !== undefined) {
        let base = TL.CONS[ch]; i++;
        if (i < cs.length && cs[i] === "़") { if (TL.NUKTA[ch] !== undefined) base = TL.NUKTA[ch]; i++; }
        U.push({ c: base, d: ch, v: null, h: false, n: "", x: "" });
      } else if (TL.VOWEL[ch] !== undefined) { U.push({ c: null, d: "", v: TL.VOWEL[ch], h: false, n: "", x: "" }); i++; }
      else if (TL.MATRA[ch] !== undefined && U.length) { U[U.length - 1].v = TL.MATRA[ch]; i++; }
      else if (ch === "्" && U.length) { U[U.length - 1].h = true; i++; }
      else if ((ch === "ं" || ch === "ँ") && U.length) { U[U.length - 1].n = ch === "ं" ? "a" : "c"; i++; }
      else if (ch === "ः" && U.length) { U[U.length - 1].x = "h"; i++; }
      else i++;
    }
    return U;
  }
  function tlWord(word) {
    word = word.normalize("NFC");
    if (Object.prototype.hasOwnProperty.call(TL.DICT, word)) return TL.DICT[word];
    const U = tlUnits(word), n = U.length;
    const cand = U.map((u) => u.c !== null && u.v === null && !u.h), keep = cand.slice();
    const hasV = (j) => U[j].v !== null || keep[j];
    for (let j = n - 1; j >= 0; j--) {
      if (!cand[j]) continue;
      if (j === n - 1) keep[j] = n === 1;
      else if (j > 0 && hasV(j - 1) && hasV(j + 1) && !U[j + 1].h) keep[j] = false;
    }
    const out = [];
    for (let j = 0; j < n; j++) {
      const u = U[j];
      if (u.c === null) out.push(u.v);
      else {
        let c = u.c; const vow = u.v !== null ? u.v : (keep[j] ? "a" : "");
        if (c === "v") c = ["a", "aa", "o", "au"].includes(vow) ? "w" : "v";
        out.push(c + vow);
      }
      if (u.n) out.push(u.n === "a" && j + 1 < n && TL.LABIAL.includes(U[j + 1].d) ? "m" : "n");
      if (u.x) out.push("h");
    }
    const s = out.join("").replace(/chchh/g, "chh");
    return s.length > 3 && s.endsWith("aa") ? s.slice(0, -2) + "a" : s;
  }
  function deva2roman(text) {
    const t = Array.from(text).map((c) => TL.DIGITS[c] !== undefined ? TL.DIGITS[c] : c).join("").replace(/[।॥]/g, ".");
    return t.replace(TL_RUN, (m) => tlWord(m));
  }
  const hasDeva = (t) => /[ऀ-ॿ]/.test(t);

  // ---------------------------------------------------------------- models
  const sigmoidNeg = (a, b, d) => 1 / (1 + Math.exp(a * d + b));      // sklearn sigmoid calibration: expit(-(a*d + b))

  function Engine(M) {
    const E = { M };
    const L = M.language; E.lang = { vec: loadVec(L.vec), classes: L.classes, folds: L.folds, scale: f32(L.scale), q: i16(L.w), b: L.b, a: L.a, c: L.c, thr: L.threshold };
    E.spam = { folds: M.spam.folds.map((f) => ({ w: loadVec(f.w), c: loadVec(f.c), scale: f.scale, coef: i16(f.coef), b: f.b, a: f.a, cb: f.cb })), thr: M.spam.thresholds };
    for (const k of ["sent_hi", "sent_en"]) {
      const S = M[k], vs = Object.values(S.vecs).map(loadVec);
      E[k] = { vs, classes: S.classes, scale: f32(S.scale), q: i16(S.w), b: S.b, nrow: S.classes.length === 2 ? 1 : S.classes.length };
    }
    for (const k of ["theme_en", "theme_hi"]) {
      const T = M[k]; E[k] = { idx: new Map(T.words.map((w, i) => [w, i])), scale: f32(T.scale), q: i8(T.vecs), dim: T.dim, idf: f32(T.idf), pc: T.pc ? Float64Array.from(T.pc) : null, centers: f32(T.centers), kk: T.k, names: T.names, labels: T.labels, rs: T.rs || null, thr: T.thr, cscale: T.cscale ? f32(T.cscale) : null, protos: T.protos ? f32(T.protos) : null, np: T.np || 0 };
    }
    return E;
  }

  function detectLanguage(E, text) {
    const m = E.lang, x = vectorize(m.vec, fixSpacing(text)), C = m.classes.length, P = new Float64Array(C);
    for (let f = 0; f < m.folds; f++) {
      const p = new Float64Array(C); let s = 0;
      for (let c = 0; c < C; c++) {
        const r = f * C + c; let d = m.b[r];
        const sc = m.scale[r], nr = m.folds * C; let acc = 0;
        for (let j = 0; j < x.idx.length; j++) acc += x.val[j] * m.q[x.idx[j] * nr + r];
        d += acc * sc;
        p[c] = sigmoidNeg(m.a[r], m.c[r], d); s += p[c];
      }
      for (let c = 0; c < C; c++) P[c] += (s ? p[c] / s : 1 / C) / m.folds;
    }
    let bi = 0; for (let c = 1; c < C; c++) if (P[c] > P[bi]) bi = c;
    return { lang: m.classes[bi], conf: P[bi], ok: P[bi] >= m.thr };
  }

  function spamScore(E, text, lang) {
    const t = normalize(text), st = structural(t), folds = E.spam.folds; let p = 0;
    for (const f of folds) {
      const xw = vectorize(f.w, t), xc = vectorize(f.c, t); let acc = 0;
      for (let j = 0; j < xw.idx.length; j++) acc += xw.val[j] * f.coef[xw.idx[j]];
      for (let j = 0; j < xc.idx.length; j++) acc += xc.val[j] * f.coef[f.w.n + xc.idx[j]];
      for (let j = 0; j < st.length; j++) acc += st[j] * f.coef[f.w.n + f.c.n + j];
      p += sigmoidNeg(f.a, f.cb, acc * f.scale + f.b) / folds.length;
    }
    const thr = E.spam.thr[lang === "hinglish" ? "hinglish" : "en"];
    return { p, thr, isSpam: p >= thr };
  }

  function sentScores(E, key, text) {
    const m = E[key], nr = m.nrow, d = new Float64Array(nr);
    for (let r = 0; r < nr; r++) d[r] = m.b[r];
    let off = 0;
    for (const v of m.vs) {
      const x = vectorize(v, text);
      for (let r = 0; r < nr; r++) { let acc = 0; for (let j = 0; j < x.idx.length; j++) acc += x.val[j] * m.q[(off + x.idx[j]) * nr + r]; d[r] += acc * m.scale[r]; }
      off += v.n;
    }
    return d;
  }
  function sentiment(E, key, text, withMargin) {
    const m = E[key], d = sentScores(E, key, text), nr = m.nrow;
    let label, margin;
    if (nr === 1) { label = m.classes[d[0] > 0 ? 1 : 0]; margin = Math.abs(d[0]); }
    else { let bi = 0; for (let r = 1; r < nr; r++) if (d[r] > d[bi]) bi = r; label = m.classes[bi]; const t = Array.from(d).sort((a, b) => a - b); margin = t[nr - 1] - t[nr - 2]; }
    return withMargin ? { label, margin } : label;
  }

  const NO_THEME = "No clear theme";
  function theme(E, key, text) {
    const T = E[key], dim = T.dim, v = new Float64Array(dim); let ws = 0, known = 0;
    for (const w of (text.toLowerCase().match(/[a-zऀ-ॿ']{2,}/g) || [])) {
      const i = T.idx.get(w); if (i === undefined) continue;
      known++; const wt = T.idf[i]; ws += wt;
      for (let k = 0; k < dim; k++) v[k] += wt * T.q[i * dim + k] * T.scale[i];
    }
    if (ws) for (let k = 0; k < dim; k++) v[k] /= ws;
    if (T.pc) { let pr = 0; for (let k = 0; k < dim; k++) pr += v[k] * T.pc[k]; for (let k = 0; k < dim; k++) v[k] -= pr * T.pc[k]; }
    let nrm = 0; for (let k = 0; k < dim; k++) nrm += v[k] * v[k];
    nrm = Math.sqrt(nrm) || 1; for (let k = 0; k < dim; k++) v[k] /= nrm;
    let best = 0, bd = Infinity;
    for (let c = 0; c < T.kk; c++) { let s = 0; for (let k = 0; k < dim; k++) { const e = v[k] - T.centers[c * dim + k]; s += e * e; } if (s < bd) { bd = s; best = c; } }
    let rejected = false, score = Math.sqrt(bd);
    if (T.rs) {
      if (T.rs === "centroid_scaled") score = Math.sqrt(bd) / T.cscale[best];
      else if (T.rs !== "centroid") {
        let m = Infinity;
        for (let p = 0; p < T.np; p++) { let s = 0; for (let k = 0; k < dim; k++) { const e = v[k] - T.protos[p * dim + k]; s += e * e; } if (s < m) m = s; }
        score = Math.sqrt(m);
      }
      rejected = known === 0 || score > T.thr;
    }
    return { cluster: best, terms: T.names[best], label: rejected ? NO_THEME : (T.labels[String(best)] || null), rejected, score };
  }

  function analyze(E, text) {
    const out = { text, needs_review: false, steps: [] }, names = E.M.meta.lang_names;
    let { lang, conf, ok } = detectLanguage(E, text);
    let modelText = text, roman = null;
    if (hasDeva(text)) {
      roman = deva2roman(text); modelText = roman;
      if (!(ok && lang === "hi")) { const d2 = detectLanguage(E, roman); if (d2.ok && (d2.lang === "en" || d2.lang === "hinglish")) { lang = d2.lang; conf = d2.conf; ok = true; } }
      out.transliterated = roman;
    }
    out.language_confidence = conf;
    if (!ok) { out.language = "uncertain"; out.language_name = `Uncertain (best guess: ${names[lang] || lang}, ${Math.round(conf * 100)}%)`; out.needs_review = true; out.spam = null; return out; }
    out.language = lang; out.language_name = names[lang] || lang;
    const mlang = lang === "hi" ? "hinglish" : lang;
    if (mlang !== "en" && mlang !== "hinglish") { out.spam = null; return out; }
    const s = spamScore(E, modelText, mlang);
    out.spam = s.isSpam ? "spam" : "ham"; out.spam_prob = s.p; out.spam_threshold = s.thr;
    if (s.isSpam) return out;
    const sr = sentiment(E, mlang === "en" ? "sent_en" : "sent_hi", modelText, true);
    out.sentiment = sr.label; out.sentiment_margin = sr.margin;
    const t = theme(E, mlang === "en" ? "theme_en" : "theme_hi", modelText);
    out.cluster = t.cluster; out.cluster_terms = t.terms; out.cluster_label = t.label; out.theme_rejected = t.rejected;
    return out;
  }

  // Which words drove the decision? Leave-one-word-out: remove each word, re-score, and see how far the score moves.
  function sentSigned(E, key, text, label) {
    const m = E[key], d = sentScores(E, key, text), nr = m.nrow;
    if (nr === 1) return m.classes[1] === label ? d[0] : -d[0];
    const li = m.classes.indexOf(label); let best = -Infinity;
    for (let r = 0; r < nr; r++) if (r !== li && d[r] > best) best = d[r];
    return d[li] - best;
  }
  function explain(E, text, a) {
    const modelText = a.transliterated || text, parts = modelText.split(/(\s+)/), idx = [];
    parts.forEach((t, i) => { if (t.trim()) idx.push(i); });
    if (idx.length < 2 || idx.length > 60 || !a.spam) return null;
    const lang = a.language === "en" ? "en" : "hinglish", spam = a.spam === "spam", key = lang === "en" ? "sent_en" : "sent_hi";
    const base = spam ? spamScore(E, modelText, lang).p : sentSigned(E, key, modelText, a.sentiment);
    const words = idx.map((i) => {
      const rest = parts.filter((_, j) => j !== i).join("").replace(/\s+/g, " ").trim();
      const v = spam ? spamScore(E, rest, lang).p : sentSigned(E, key, rest, a.sentiment);
      return { w: parts[i], d: base - v };            // positive = this word pushed towards the decision
    });
    return { kind: spam ? "spam" : "sentiment", words, base };
  }

  const api = { explain, Engine, analyze, deva2roman, hasDeva, sentScores, detectLanguage, spamScore, sentiment, theme, normalize, fixSpacing, structural, vectorize };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.MIEngine = api;
})(typeof self !== "undefined" ? self : this);
