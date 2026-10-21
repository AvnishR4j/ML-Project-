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
  function loadVec(v) {
    return { an: v.an, ng: v.ng, prep: prepFns[v.prep], idx: new Map(v.vocab.map((t, i) => [t, i])), idf: f32(v.idf), n: v.vocab.length };
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
      const T = M[k]; E[k] = { idx: new Map(T.words.map((w, i) => [w, i])), scale: f32(T.scale), q: i8(T.vecs), dim: T.dim, idf: f32(T.idf), centers: f32(T.centers), kk: T.k, names: T.names, labels: T.labels };
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

  function sentiment(E, key, text) {
    const m = E[key], nr = m.nrow, d = new Float64Array(nr);
    for (let r = 0; r < nr; r++) d[r] = m.b[r];
    let off = 0;
    for (const v of m.vs) {
      const x = vectorize(v, text);
      for (let r = 0; r < nr; r++) { let acc = 0; for (let j = 0; j < x.idx.length; j++) acc += x.val[j] * m.q[(off + x.idx[j]) * nr + r]; d[r] += acc * m.scale[r]; }
      off += v.n;
    }
    if (nr === 1) return m.classes[d[0] > 0 ? 1 : 0];
    let bi = 0; for (let r = 1; r < nr; r++) if (d[r] > d[bi]) bi = r;
    return m.classes[bi];
  }

  function theme(E, key, text) {
    const T = E[key], dim = T.dim, v = new Float64Array(dim); let ws = 0;
    for (const w of (text.toLowerCase().match(/[a-zऀ-ॿ']{2,}/g) || [])) {
      const i = T.idx.get(w); if (i === undefined) continue;
      const wt = T.idf[i]; ws += wt;
      for (let k = 0; k < dim; k++) v[k] += wt * T.q[i * dim + k] * T.scale[i];
    }
    let nrm = 0; if (ws) for (let k = 0; k < dim; k++) { v[k] /= ws; nrm += v[k] * v[k]; }
    nrm = Math.sqrt(nrm) || 1; for (let k = 0; k < dim; k++) v[k] /= nrm;
    let best = 0, bd = Infinity;
    for (let c = 0; c < T.kk; c++) { let s = 0; for (let k = 0; k < dim; k++) { const e = v[k] - T.centers[c * dim + k]; s += e * e; } if (s < bd) { bd = s; best = c; } }
    return { cluster: best, terms: T.names[best], label: T.labels[String(best)] || null };
  }

  function analyze(E, text) {
    const out = { text, needs_review: false, steps: [] }, names = E.M.meta.lang_names;
    const { lang, conf, ok } = detectLanguage(E, text);
    out.language_confidence = conf;
    if (!ok) { out.language = "uncertain"; out.language_name = `Uncertain (best guess: ${names[lang] || lang}, ${Math.round(conf * 100)}%)`; out.needs_review = true; out.spam = null; return out; }
    out.language = lang; out.language_name = names[lang] || lang;
    if (lang !== "en" && lang !== "hinglish") { out.spam = null; return out; }
    const s = spamScore(E, text, lang);
    out.spam = s.isSpam ? "spam" : "ham"; out.spam_prob = s.p; out.spam_threshold = s.thr;
    if (s.isSpam) return out;
    out.sentiment = sentiment(E, lang === "en" ? "sent_en" : "sent_hi", text);
    const t = theme(E, lang === "en" ? "theme_en" : "theme_hi", text);
    out.cluster = t.cluster; out.cluster_terms = t.terms; out.cluster_label = t.label;
    return out;
  }

  const api = { Engine, analyze, detectLanguage, spamScore, sentiment, theme, normalize, fixSpacing, structural, vectorize };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.MIEngine = api;
})(typeof self !== "undefined" ? self : this);
