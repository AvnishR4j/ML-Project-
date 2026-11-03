/* Book mode. Re-seats the five sections of the page as the pages of a real book and turns them with a bending page.
   The bend follows the same arc as the 3D sketchbook on avnish-workspace.vercel.app (scene.ts, deformPage):
       angle(u) = progress * PI - sin(u * PI) * bend,   bend = sin(progress * PI) * 0.62,   u = 0 at the spine, 1 at the page edge
   Each moving page is cut into strips of the live page (DOM clones), which are placed on that arc in CSS 3D.
   At rest the pages are the live page, so the demo, charts, filters and the file upload all keep working. */
(function () {
  'use strict';
  var root = document.documentElement;
  if (!root.getAttribute('data-book')) return;
  var $ = function (id) { return document.getElementById(id); };
  var q1 = function (sel, ctx) { return (ctx || document).querySelector(sel); };
  var qa = function (sel, ctx) { return [].slice.call((ctx || document).querySelectorAll(sel)); };
  var el = function (tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  var clamp = function (x, a, b) { return Math.max(a, Math.min(b, x)); };

  // ---------------------------------------------------------------- gather the old page's pieces
  var hero = q1('header.hero'), s3 = $('s3'), s3b = $('s3b'), s2 = $('s2'), s4 = $('s4'), s5 = $('s5'), s6 = $('s6'), footer = q1('footer');
  var five = q1('.five', hero); var fiveHTML = five ? five.outerHTML : '';

  function mk(side, cls) { var p = el('section', 'pg ' + side + (cls ? ' ' + cls : '')); var inn = el('div', 'pgin'); p.appendChild(inn); p._in = inn; return p; }
  function put(page) { for (var i = 1; i < arguments.length; i++) if (arguments[i]) page._in.appendChild(arguments[i]); }
  function wrap(cls) { var d = el('div', cls); for (var i = 1; i < arguments.length; i++) if (arguments[i]) d.appendChild(arguments[i]); return d; }

  var PG = [];
  // 0 cover
  var cover = mk('R', 'cover'); put(cover, hero); cover._in.appendChild(el('div', 'cover-foot', 'Avnish Raj &nbsp;·&nbsp; open me'));
  // 1 + 2  Demo (one message, or a whole file)
  var d1 = mk('L'), d2 = mk('R');
  var modes = el('div', 'modes', '<button type="button" data-m="single" aria-pressed="true">One message</button><button type="button" data-m="file" aria-pressed="false">A whole file</button>');
  var demo = q1('.demo', s3);
  put(d1, modes, wrap('only-single', q1('.sh', s3), q1('.a', demo)));
  var batchHead = q1('.sh', s3b), batchRow = q1('.row2', s3b), batchProg = $('b-prog'), batchNote = $('b-note');
  put(d1, wrap('batch only-file', batchHead, batchRow, batchProg, batchNote));
  put(d2, wrap('only-single', q1('.b', demo)));
  var bout = $('b-out'); put(d2, wrap('batch only-file', bout, el('p', 'ph', 'Your results will appear here.')));
  // 3 + 4  Pipeline
  var p1 = mk('L'), p2 = mk('R'); var cells = qa('.cell', s2);
  put(p1, q1('.sh', s2), wrap('cells', cells[0], cells[1]));
  put(p2, wrap('cells', cells[2], cells[3]), q1('.two2', s2));
  // 5 + 6  Performance
  var f1 = mk('L'), f2 = mk('R'), blks = qa('.blk', s4);
  put(f1, q1('.sh', s4), q1('.stats.first', s4), q1('.stats:not(.first)', s4), blks[0]);
  put(f2, blks[1], blks[2]);
  // 7 + 8  Robustness
  var r1 = mk('L'), r2 = mk('R');
  put(r1, q1('.sh', s5), q1('.rob .a', s5)); put(r2, q1('.rob .b', s5));
  // 9 + 10 Explorer
  var e1 = mk('L'), e2 = mk('R');
  put(e1, q1('.sh', s6), q1('.bar', s6), $('count'));
  put(e2, q1('.tbox', s6));
  // 11 colophon (inside of the back cover's board)
  var end = mk('L'); end._in.innerHTML = '<div class="colo"><h2 class="d">The end</h2>' + fiveHTML.replace('class="five in1"', 'class="five"') +
    '<p style="margin-top:26px">' + (footer ? footer.innerHTML : '') + '</p>' +
    '<p style="margin-top:18px"><a href="https://github.com/AvnishR4j/ML-Project-" target="_blank" rel="noopener">Code and README on GitHub</a> &nbsp;·&nbsp; <a href="classic.html">Scroll version of this page</a></p></div>';
  PG = [cover, d1, d2, p1, p2, f1, f2, r1, r2, e1, e2, end];

  // ---------------------------------------------------------------- build the book
  var desk = el('div', 'desk'), bk = el('div', '', ''); bk.id = 'bk'; bk.setAttribute('data-demo', 'single');
  var pages = el('div', 'pages');
  bk.appendChild(el('div', 'board board-l')); bk.appendChild(el('div', 'board board-r')); bk.appendChild(el('div', 'stk stk-l')); bk.appendChild(el('div', 'stk stk-r'));
  bk.appendChild(pages);
  PG.forEach(function (p, i) {
    p.classList.add('off'); p.inert = true; p.setAttribute('aria-hidden', 'true'); p._i = i;
    if (i > 0) p.appendChild(el('span', 'pn d', String(i))); pages.appendChild(p);
  });
  var cast = el('div', 'cast'); bk.appendChild(cast);
  var NAMES = ['Cover', 'Demo', 'Pipeline', 'Performance', 'Robustness', 'Explorer', 'End'], SLUG = ['cover', 'demo', 'pipeline', 'performance', 'robustness', 'explorer', 'end'];
  var tabs = el('div', 'btabs'); tabs.setAttribute('role', 'tablist');
  for (var t = 1; t <= 5; t++) { var b = el('button', 'bt2 d', NAMES[t]); b.type = 'button'; b.dataset.s = t; b.setAttribute('aria-label', 'Open ' + NAMES[t]); tabs.appendChild(b); }
  bk.appendChild(tabs);
  desk.appendChild(bk); document.body.appendChild(desk);
  var hint = el('div', 'dhint', 'drag a page to turn it &nbsp;·&nbsp; ← → keys &nbsp;·&nbsp; <a href="classic.html">scroll version</a>'); document.body.appendChild(hint);

  // ---------------------------------------------------------------- size
  var PW = 600, PH = 760;
  function size() {
    var W = innerWidth - 150, H = innerHeight - 96, ratio = 0.8;
    PW = Math.floor(Math.min(W / 2, H * ratio, 820)); PH = Math.floor(PW / ratio);
    bk.style.setProperty('--pw', PW + 'px'); bk.style.setProperty('--ph', PH + 'px'); var th = Math.floor((PH - 60) / 5 - 7); bk.style.setProperty('--th', th + 'px'); bk.style.setProperty('--tf', Math.min(17, Math.max(12, Math.floor(th / 6.2))) + 'px'); cover._in.style.zoom = Math.min(1, PH / 760);
  }
  size();

  // ---------------------------------------------------------------- state
  var s = 0, target = 0, anim = null;                       // s = number of leaves turned: 0 = closed cover, 1..5 = spreads, 6 = end
  function leftOf(sp) { return sp >= 1 ? PG[2 * sp - 1] : null; }
  function rightOf(sp) { return sp <= 5 ? PG[2 * sp] : null; }
  function setPage(p, side) {
    if (!p) return;
    p.classList.remove('off', 'L', 'R'); p.classList.add(side); p.inert = false; p.removeAttribute('aria-hidden');
    p.style.visibility = '';
  }
  function hideAll() { PG.forEach(function (p) { p.classList.add('off'); p.inert = true; p.setAttribute('aria-hidden', 'true'); }); }
  function showSpread(sp, extraL, extraR) {
    hideAll(); setPage(leftOf(sp), 'L'); setPage(rightOf(sp), 'R'); setPage(extraL, 'L'); setPage(extraR, 'R');
    // pages shown only as the surface under a turning page keep their own side classes
  }
  function setV(v) { bk.style.setProperty('--v', v.toFixed(4)); }
  var sections = { 1: 's3', 2: 's2', 3: 's4', 4: 's5', 5: 's6' };
  function opened(sp) {
    bk.dataset.spread = sp;
    qa('.bt2', tabs).forEach(function (b) { b.setAttribute('aria-current', String(+b.dataset.s === sp)); });
    if (sections[sp] && window.__reveal) window.__reveal(sections[sp]);
    var h = '#' + SLUG[sp]; if (location.hash !== h) { try { history.replaceState(null, '', sp === 0 ? location.pathname + location.search : h); } catch (e) {} }
    document.title = (sp >= 1 && sp <= 5 ? NAMES[sp] + ' · ' : '') + 'Multilingual Message Intelligence';
  }

  // ---------------------------------------------------------------- the turning page
  var N = 22, W_STRIP = 0, strips = [], turner = null, reduceMotion = false;
  function lightClone(page) {
    var c = page.cloneNode(true); c.className = 'pg pgc ' + (page.classList.contains('R') ? 'R' : (page.classList.contains('L') ? 'L' : (page._i % 2 ? 'L' : 'R'))) + (page.classList.contains('cover') ? ' cover' : '');
    c.removeAttribute('aria-hidden'); c.inert = false; c.style.display = '';
    qa('[id]', c).forEach(function (n) { n.removeAttribute('id'); });
    qa('tbody', c).forEach(function (tb) { while (tb.rows.length > 24) tb.deleteRow(tb.rows.length - 1); });
    qa('canvas', c).forEach(function (cv) { cv.parentNode.removeChild(cv); });
    c.style.width = PW + 'px'; c.style.height = PH + 'px';
    var st = page._in.scrollTop;
    if (st > 0) { var inn = c._in || q1('.pgin', c); inn.style.overflow = 'hidden'; var w = el('div'); while (inn.firstChild) w.appendChild(inn.firstChild); w.style.transform = 'translateY(' + (-st) + 'px)'; inn.appendChild(w); }
    return c;
  }
  function beginTurn(leaf, forward) {
    var front = PG[2 * leaf], back = PG[2 * leaf + 1], underR = forward ? PG[2 * leaf + 2] : null, underL = forward ? null : PG[2 * leaf - 1];
    var fc = lightClone(front), bc = lightClone(back);
    // surface under the moving page
    showSpread(forward ? s : s, null, null);
    if (forward) { if (underR) setPage(underR, 'R'); } else { if (underL) setPage(underL, 'L'); }
    front.style.visibility = 'hidden'; back.style.visibility = 'hidden';
    if (!forward) { back.classList.remove('off'); back.style.visibility = 'hidden'; setPage(back, 'L'); back.style.visibility = 'hidden'; }
    if (forward) { setPage(front, 'R'); front.style.visibility = 'hidden'; }
    turner = el('div', 'turner'); strips = []; W_STRIP = PW / N;
    for (var k = 0; k < N; k++) {
      var st = el('div', 'strip'); st.style.width = (W_STRIP + 0.6) + 'px';
      var f = el('div', 'sf'), bkf = el('div', 'sf sback');
      var hf = k === 0 ? fc : fc.cloneNode(true), hb = k === 0 ? bc : bc.cloneNode(true);
      hf.style.left = (-k * W_STRIP) + 'px'; hb.style.left = (-(PW - (k + 1) * W_STRIP)) + 'px';
      var sh1 = el('i'), sh2 = el('i'); f.appendChild(hf); f.appendChild(sh1); bkf.appendChild(hb); bkf.appendChild(sh2);
      st.appendChild(f); st.appendChild(bkf); st._sh = [sh1, sh2]; turner.appendChild(st); strips.push(st);
    }
    bk.appendChild(turner);
    anim = { leaf: leaf, forward: forward, p: 0, front: front, back: back, underR: underR, underL: underL };
    renderTurn(0);
  }
  function renderTurn(p) {
    var a = anim; a.p = p;
    var forward = a.forward, turned = forward ? p : 1 - p;               // 0 = leaf on the right, 1 = leaf on the left
    var bend = Math.sin(turned * Math.PI) * 0.62, xs = [], zs = [], maxX = -1e9, minX = 1e9;
    for (var k = 0; k <= N; k++) {
      var u = k / N, d = u * PW, ang = turned * Math.PI - Math.sin(u * Math.PI) * bend;
      var lift = Math.sin(u * Math.PI) * Math.sin(turned * Math.PI) * PW * 0.07;     // paper rises a little in the middle of the turn
      xs.push(Math.cos(ang) * d); zs.push(Math.sin(ang) * d + lift); maxX = Math.max(maxX, xs[k]); minX = Math.min(minX, xs[k]);
    }
    for (var i = 0; i < N; i++) {
      var dx = xs[i + 1] - xs[i], dz = zs[i + 1] - zs[i], th = Math.atan2(dz, dx), len = Math.hypot(dx, dz);
      var st = strips[i];
      st.style.transform = 'translate3d(' + xs[i].toFixed(2) + 'px,0,' + zs[i].toFixed(2) + 'px) rotateY(' + (-th).toFixed(4) + 'rad) scaleX(' + (len / W_STRIP).toFixed(4) + ')';
      var tilt = Math.abs(Math.sin(th));                                // edge-on strips are darker
      var base = 0.05 + 0.5 * tilt * tilt + 0.1 * Math.sin(turned * Math.PI) * (i / N);
      st._sh[0].style.opacity = base.toFixed(3); st._sh[1].style.opacity = base.toFixed(3);
    }
    // shadow cast on the page below
    var right = maxX > 0 && forward || (!forward && maxX > 0);
    var strength = Math.sin(turned * Math.PI);
    if (turned < 0.5) { cast.style.left = '50%'; cast.style.width = clamp(maxX, 0, PW) + 40 + 'px'; cast.style.background = 'linear-gradient(90deg,rgba(40,20,0,.0),rgba(40,20,0,.30) 70%,rgba(40,20,0,.0))'; }
    else { var w = clamp(-minX, 0, PW) + 40; cast.style.left = 'calc(50% - ' + w + 'px)'; cast.style.width = w + 'px'; cast.style.background = 'linear-gradient(90deg,rgba(40,20,0,.0),rgba(40,20,0,.30) 30%,rgba(40,20,0,.0))'; }
    cast.style.opacity = (strength * 0.9).toFixed(3);
    var v = a.leaf + turned; setV(v);
  }
  function endTurn(commit) {
    var a = anim; if (!a) return;
    if (turner && turner.parentNode) turner.parentNode.removeChild(turner);
    turner = null; strips = []; cast.style.opacity = 0;
    if (commit) s += a.forward ? 1 : -1;
    a.front.style.visibility = ''; a.back.style.visibility = '';
    showSpread(s); setV(s); anim = null; opened(s);
  }

  // ---------------------------------------------------------------- animation driver
  var raf = 0;
  function runTo(goal, dur, done) {
    var from = anim.p, t0 = performance.now(), a = anim;
    if (reduceMotion) dur = 40;
    cancelAnimationFrame(raf);
    (function tick(now) {
      var t = Math.min(1, (now - t0) / dur), e = t * t * (3 - 2 * t);
      renderTurn(from + (goal - from) * e);
      if (t < 1) raf = requestAnimationFrame(tick); else { endTurn(goal === 1); if (done) done(); pump(); }
    })(t0);
  }
  function turn(forward, dur) {
    if (anim) return;
    var leaf = forward ? s : s - 1; if (leaf < 0 || leaf > 5) return;
    beginTurn(leaf, forward);
    requestAnimationFrame(function () { if (anim) runTo(1, dur || 780); });
  }
  function pump() { if (anim) return; if (target > s) turn(true, Math.abs(target - s) > 1 ? 380 : 780); else if (target < s) turn(false, Math.abs(target - s) > 1 ? 380 : 780); }
  function goTo(n) { target = clamp(n, 0, 6); pump(); }

  // ---------------------------------------------------------------- drag / swipe / keys / tabs
  var drag = null, justDragged = false, NODRAG = 'textarea,input,select,.tbox,.tw,.cvs';
  bk.addEventListener('pointerdown', function (e) {
    if (anim || e.button > 0 || (e.target.closest && e.target.closest(NODRAG))) { drag = null; return; }
    if (e.target.closest && e.target.closest('.bt2')) { drag = null; return; }
    drag = { x: e.clientX, y: e.clientY, active: false, id: e.pointerId };
  });
  bk.addEventListener('pointermove', function (e) {
    if (!drag) return;
    var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (!drag.active) {
      if (Math.abs(dx) < 12 || Math.abs(dx) < Math.abs(dy) * 1.2) return;
      var fwd = dx < 0; if ((fwd && s >= 6) || (!fwd && s <= 0)) { drag = null; return; }
      drag.active = true; drag.fwd = fwd; target = s + (fwd ? 1 : -1);
      try { bk.setPointerCapture(drag.id); } catch (er) {}
      beginTurn(fwd ? s : s - 1, fwd);
      getSelection && getSelection().removeAllRanges && getSelection().removeAllRanges();
    }
    var span = Math.min(350, PW * 0.62);
    renderTurn(clamp((drag.fwd ? -dx : dx) / span, 0, 0.99));
  });
  function release(e) {
    if (!drag) return;
    var d = drag; drag = null;
    if (!d.active) return;
    justDragged = true; setTimeout(function () { justDragged = false; }, 60);
    var p = anim.p, ok = p > 0.22, dur = 780 * (0.4 + 0.6 * Math.abs((ok ? 1 : 0) - p));
    if (!ok) target = s;
    runTo(ok ? 1 : 0, dur);
  }
  bk.addEventListener('pointerup', release); bk.addEventListener('pointercancel', release);
  document.addEventListener('click', function (e) { if (justDragged) { e.stopPropagation(); e.preventDefault(); } }, true);
  document.addEventListener('keydown', function (e) {
    var t = e.target, tag = t && t.tagName; if (tag === 'TEXTAREA' || tag === 'INPUT' || tag === 'SELECT' || (t && t.isContentEditable) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'ArrowRight' || e.key === 'PageDown') { goTo(target + 1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { goTo(target - 1); e.preventDefault(); }
    else if (e.key === 'Home') goTo(0); else if (e.key === 'End') goTo(6);
  });
  tabs.addEventListener('click', function (e) { var b = e.target.closest('.bt2'); if (b) goTo(+b.dataset.s); });
  cover.addEventListener('click', function (e) { if (s === 0 && !anim && !(e.target.closest && e.target.closest('a,button'))) goTo(1); });
  modes.addEventListener('click', function (e) { var b = e.target.closest('button'); if (!b) return; bk.setAttribute('data-demo', b.dataset.m); qa('button', modes).forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); }); });
  // choosing a file or the sample switches the demo page to the file view
  ['b-file', 'b-sample', 'b-go'].forEach(function (id) { var n = $(id); if (n) n.addEventListener('click', function () { bk.setAttribute('data-demo', 'file'); qa('button', modes).forEach(function (x) { x.setAttribute('aria-pressed', String(x.dataset.m === 'file')); }); }, true); });
  window.addEventListener('resize', function () { size(); if (anim) { endTurn(false); } });
  window.addEventListener('hashchange', function () { var i = SLUG.indexOf(location.hash.slice(1)); if (i >= 0) goTo(i); });

  // ---------------------------------------------------------------- start
  var start = Math.max(0, SLUG.indexOf(location.hash.slice(1)));
  s = target = start; showSpread(s); setV(s); opened(s);
  try { reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e) {}
})();
