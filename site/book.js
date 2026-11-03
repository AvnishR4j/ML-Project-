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
  var cover = mk('R', 'cover'); put(cover, hero); cover._in.appendChild(el('div', 'cover-foot', 'open me'));
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
    hideAll(); setPage(leftOf(sp), 'L'); setPage(rightOf(sp), 'R'); shown = [leftOf(sp), rightOf(sp)];
  }
  // only the four elements that depend on how far the book is open are touched (a CSS variable on #bk would restyle every page each frame)
  var curV = 0, boardL = q1('.board-l', bk), stkL = q1('.stk-l', bk), stkR = q1('.stk-r', bk);
  function setV(v) {
    curV = v; var t = 'translateX(' + (-0.5 * PW * (1 - clamp(v, 0, 1))).toFixed(1) + 'px)'; if (bk._t !== t) { bk._t = t; bk.style.transform = t; }
    var ol = clamp(v, 0, 1), or = clamp(6 - v, 0, 1); if (boardL._o !== ol) { boardL._o = ol; boardL.style.opacity = stkL.style.opacity = ol; } if (stkR._o !== or) { stkR._o = or; stkR.style.opacity = or; }
    var wr = Math.round(2 + Math.max(0, (5.5 - v) * 2.4)), wl = Math.round(2 + Math.max(0, (v - 0.5) * 2.4));
    if (stkR._w !== wr) { stkR._w = wr; stkR.style.width = 'calc(50% + ' + wr + 'px)'; }
    if (stkL._w !== wl) { stkL._w = wl; stkL.style.width = 'calc(50% + ' + wl + 'px)'; }
  }
  var sections = { 1: 's3', 2: 's2', 3: 's4', 4: 's5', 5: 's6' };
  function opened(sp) {
    bk.dataset.spread = sp;
    if (window.__heroHold) window.__heroHold(sp !== 0);
    qa('.bt2', tabs).forEach(function (b) { b.setAttribute('aria-current', String(+b.dataset.s === sp)); });
    if (sections[sp] && window.__reveal) window.__reveal(sections[sp]);
    var h = '#' + SLUG[sp]; if (location.hash !== h) { try { history.replaceState(null, '', sp === 0 ? location.pathname + location.search : h); } catch (e) {} }
    document.title = (sp >= 1 && sp <= 5 ? NAMES[sp] + ' · ' : '') + 'Multilingual Message Intelligence';
  }

  // ---------------------------------------------------------------- the turning page
  // How the portfolio sketchbook stays smooth: the pages are pictures, and only one bent sheet is drawn by the graphics card.
  // Here every page is photographed into an image ahead of time (and again whenever its content changes), and the
  // leaf that turns is a lit WebGL mesh bent with the portfolio's own formula (scene.ts, deformPage). Nothing in the
  // page itself is touched while it moves, so the turn costs one small draw call per frame.
  var reduceMotion = false, MS = window.modernScreenshot;
  var SCALE = Math.min(2, window.devicePixelRatio || 1);

  // ---- page photographs
  var shots = [];                                   // per page: { cv: canvas, dirty: bool, busy: promise, gen: n }
  PG.forEach(function (p, i) { shots[i] = { cv: null, dirty: true, busy: null, gen: 0 }; });
  function sideOf(i) { return i === 0 ? 'R' : (i % 2 ? 'L' : 'R'); }
  function snap(i) {
    var sh = shots[i];
    if (sh.busy) return sh.busy;
    if (!MS) return Promise.reject(new Error('no snapshot library'));
    var p = PG[i], wasOff = p.classList.contains('off'), wasInert = p.inert;
    if (wasOff) { p.classList.remove('off'); p.classList.add('snapping', sideOf(i)); }   // laid out off-screen so it can be photographed
    var gen = ++sh.gen; sh.dirty = false;
    var tS = performance.now();
    sh.busy = MS.domToCanvas(p, {
      width: PW, height: PH, scale: SCALE, backgroundColor: null,
      style: { left: '0px', top: '0px', position: 'static', visibility: 'visible', transform: 'none' },
      filter: function (n) { return !(n.tagName === 'TR' && n.parentNode && n.parentNode.tagName === 'TBODY' && n.sectionRowIndex > 30); },
      features: { restoreScrollPosition: true, copyScrollbar: false }, timeout: 8000
    }).then(function (cv) {
      // a browser that cannot draw the page into a picture returns an empty one; treat that as a failure so the plain flip is used instead
      var g = cv.getContext('2d'), seen = 0;
      [[0.5, 0.1], [0.5, 0.5], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]].forEach(function (q) { if (g.getImageData(Math.floor(cv.width * q[0]), Math.floor(cv.height * q[1]), 1, 1).data[3] > 0) seen++; });
      if (!seen) throw new Error('blank page picture');
      if (window.__dbg) console.log('snap', i, cv.width, cv.height);
      if (PERF) PERF.snap(i, performance.now() - tS);
      if (gen === sh.gen) { sh.cv = cv; sh.tex = null; }
      return cv;
    }).finally(function () {
      sh.busy = null;
      if (p.classList.contains('snapping')) { p.classList.remove('snapping'); if (!isShown(p)) { p.classList.add('off'); p.inert = true; } }
    });
    return sh.busy;
  }
  var shown = [];
  function isShown(p) { return shown.indexOf(p) >= 0; }
  function ready(i) { var sh = shots[i]; return sh.cv && !sh.dirty ? Promise.resolve(sh.cv) : (sh.busy && !sh.dirty ? sh.busy : snap(i)); }
  // keep photographs fresh: any change inside a page marks it stale, and stale pages near the open spread are re-shot when the browser is idle
  var idle = window.requestIdleCallback || function (f) { return setTimeout(function () { f({ timeRemaining: function () { return 8; } }); }, 120); };
  var queued = false;
  function nearby() { var l = []; for (var k = 2 * s - 2; k <= 2 * s + 3; k++) if (k >= 0 && k < PG.length) l.push(k); l.sort(function (a, b) { return Math.abs(a - (2 * s)) - Math.abs(b - (2 * s)); }); return l; }
  function pumpShots() {
    if (queued) return; queued = true;
    idle(function () {
      queued = false; if (anim) { setTimeout(pumpShots, 300); return; }
      var list = nearby().concat(PG.map(function (p, i) { return i; }));
      for (var k = 0; k < list.length; k++) { var i = list[k]; if (i === 0 && s === 0 && shots[0].cv) continue; if ((shots[i].dirty || !shots[i].cv) && !shots[i].busy) { snap(i).then(pumpShots, function () {}); return; } }
      preUpload();
    }, { timeout: 4000 });
  }
  var stale = function (i) { return function (ev) { if (window.__dbg && !shots[i].dirty) console.log('dirty', i, ev && ev.type || 'mut'); shots[i].dirty = true; clearTimeout(shots[i].t); shots[i].t = setTimeout(pumpShots, 900); }; };
  if (window.MutationObserver) PG.forEach(function (p, i) {
    var mark = stale(i);
    new MutationObserver(function (recs) {
      for (var r = 0; r < recs.length; r++) { var a = recs[r]; if (a.type === 'attributes' && a.target === p) continue; if (window.__dbg && !shots[i].dirty) console.log('mut', i, a.type, a.attributeName, a.target.className || a.target.nodeName); mark(); return; }
    }).observe(p, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ['class', 'style', 'hidden', 'value', 'open', 'aria-pressed'] });
    p.addEventListener('scroll', mark, true); p.addEventListener('input', mark, true); p.addEventListener('change', mark, true);
  });
  new MutationObserver(function () { PG.forEach(function (p, i) { shots[i].dirty = true; }); pumpShots(); }).observe(bk, { attributes: true, attributeFilter: ['data-demo'] });

  // ---- WebGL leaf
  var gl = null, glc = null, prog, buf, ibuf, uni = {}, GX = 40, GY = 12, nIdx = 0, pos, nor;
  function initGL() {
    if (gl !== null) return !!gl;
    glc = el('canvas', 'leafgl'); document.body.appendChild(glc);
    try { gl = glc.getContext('webgl', { alpha: true, antialias: true, premultipliedAlpha: true, preserveDrawingBuffer: false }); } catch (e) { gl = null; }
    if (!gl) { gl = false; glc.remove(); return false; }
    var vs = 'attribute vec3 aP;attribute vec3 aN;attribute vec2 aU;uniform vec2 uO;uniform vec2 uH;uniform float uD;varying vec2 vU;varying vec3 vN;varying float vZ;' +
      'void main(){vU=aU;vN=aN;vZ=aP.z;float w=(uD-aP.z)/uD;vec2 s=uO+(aP.xy-uO)/w;vec2 c=(s-uH)/uH;gl_Position=vec4(c.x*w,-c.y*w,clamp(-aP.z/(uD*2.0),-0.99,0.99)*w,w);}';
    var fs = 'precision mediump float;uniform sampler2D uF;uniform sampler2D uB;uniform float uLift;varying vec2 vU;varying vec3 vN;varying float vZ;' +
      'void main(){vec3 n=normalize(vN);vec4 c;' +
      'if(gl_FrontFacing){c=texture2D(uF,vU);}else{c=texture2D(uB,vec2(1.0-vU.x,vU.y));n=-n;}' +
      'vec3 L=normalize(vec3(-0.35,-0.45,0.82));float dif=max(dot(n,L),0.0);' +
      'vec3 H=normalize(L+vec3(0.0,0.0,1.0));float sp=pow(max(dot(n,H),0.0),36.0)*0.10*uLift;' +
      'float lit=(0.50+0.52*dif)/0.9264;' +
      'gl_FragColor=vec4(c.rgb*lit+sp,1.0)*c.a;}';
    function sh(t, src) { var o = gl.createShader(t); gl.shaderSource(o, src); gl.compileShader(o); if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(o)); return o; }
    try {
      prog = gl.createProgram(); gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link');
    } catch (e) { gl = false; glc.remove(); return false; }
    gl.useProgram(prog);
    ['uO', 'uH', 'uD', 'uF', 'uB', 'uLift'].forEach(function (n) { uni[n] = gl.getUniformLocation(prog, n); });
    var nv = (GX + 1) * (GY + 1); pos = new Float32Array(nv * 3); nor = new Float32Array(nv * 3);
    var uv = new Float32Array(nv * 2), idx = new Uint16Array(GX * GY * 6), k = 0;
    for (var iy = 0; iy <= GY; iy++) for (var ix = 0; ix <= GX; ix++) { var v = iy * (GX + 1) + ix; uv[v * 2] = ix / GX; uv[v * 2 + 1] = iy / GY; }
    for (iy = 0; iy < GY; iy++) for (ix = 0; ix < GX; ix++) { var a = iy * (GX + 1) + ix, b = a + 1, c = a + GX + 1, d = c + 1; idx[k++] = a; idx[k++] = c; idx[k++] = b; idx[k++] = b; idx[k++] = c; idx[k++] = d; }
    nIdx = k;
    buf = { p: gl.createBuffer(), n: gl.createBuffer(), u: gl.createBuffer() };
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.u); gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
    ibuf = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibuf); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    var locs = { aP: gl.getAttribLocation(prog, 'aP'), aN: gl.getAttribLocation(prog, 'aN'), aU: gl.getAttribLocation(prog, 'aU') };
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.u); gl.enableVertexAttribArray(locs.aU); gl.vertexAttribPointer(locs.aU, 2, gl.FLOAT, false, 0, 0);
    buf.locs = locs;
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    glc.addEventListener('webglcontextlost', function (e) { e.preventDefault(); gl = false; }, false);
    return true;
  }
  function upload(t, cv) {
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cv);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }
  var texc = {};                                    // page index -> GPU texture of its current photograph
  function texFor(i) {
    var sh = shots[i]; if (!sh.cv) return null;
    var e = texc[i]; if (e && e.cv === sh.cv) return e.t;
    if (!e) e = texc[i] = { t: gl.createTexture(), cv: null };
    upload(e.t, sh.cv); e.cv = sh.cv; return e.t;
  }
  function preUpload() {
    if (!initGL() || anim) return;
    [2 * s, 2 * s + 1, 2 * s - 2, 2 * s - 1].forEach(function (i) { if (i >= 0 && i < PG.length && shots[i].cv && !shots[i].dirty) texFor(i); });
    var keep = nearby(); Object.keys(texc).forEach(function (k) { if (keep.indexOf(+k) < 0) { gl.deleteTexture(texc[k].t); delete texc[k]; } });
  }
  function warm() { [2 * s, 2 * s + 1, 2 * s - 2, 2 * s - 1].forEach(function (i) { if (i >= 0 && i < PG.length && (shots[i].dirty || !shots[i].cv) && !shots[i].busy) snap(i).catch(function () {}); }); }
  var VW = 0, VH = 0;
  function sizeGL() {
    VW = innerWidth; VH = innerHeight;
    var w = Math.round(VW * SCALE), h = Math.round(VH * SCALE);
    if (glc.width !== w || glc.height !== h) { glc.width = w; glc.height = h; }
    gl.viewport(0, 0, w, h);
  }

  // ---- geometry: the portfolio's bend, in screen pixels (x right, y down, z towards the viewer)
  var geo = { spine0: 0, top: 0, v0: 0 };
  function tx(v) { return -0.5 * PW * (1 - clamp(v, 0, 1)); }      // the closed book sits centred; it slides as the cover opens
  function bendLeaf(turned) {
    var spine = geo.spine0 + tx(turned + anim.leaf) - tx(geo.v0), top = geo.top;
    var bend = Math.sin(turned * Math.PI) * 0.62, rise = Math.sin(turned * Math.PI);
    var u, d, ang, x, z, k = 0, sag;
    for (var iy = 0; iy <= GY; iy++) {
      var fy = iy / GY - 0.5;
      for (var ix = 0; ix <= GX; ix++) {
        u = ix / GX; d = (0.013 + u * 0.987) * PW;
        ang = turned * Math.PI - Math.sin(u * Math.PI) * bend;
        sag = Math.sin(u * Math.PI);
        x = Math.cos(ang) * d;
        z = Math.sin(ang) * d + sag * (0.029 + rise * 0.058) * PW;       // the sheet lifts off the block in the middle of the turn
        pos[k] = spine + x; pos[k + 1] = top + (fy * (1 + sag * rise * 0.028) + 0.5) * PH; pos[k + 2] = z; k += 3;
      }
    }
    // normals from neighbouring vertices
    var R = GX + 1;
    for (iy = 0; iy <= GY; iy++) for (ix = 0; ix <= GX; ix++) {
      var i0 = iy * R + ix, ia = iy * R + Math.min(GX, ix + 1), ib = iy * R + Math.max(0, ix - 1), ic = Math.min(GY, iy + 1) * R + ix, id = Math.max(0, iy - 1) * R + ix;
      var ax = pos[ia * 3] - pos[ib * 3], ay = pos[ia * 3 + 1] - pos[ib * 3 + 1], az = pos[ia * 3 + 2] - pos[ib * 3 + 2];
      var bx = pos[ic * 3] - pos[id * 3], by = pos[ic * 3 + 1] - pos[id * 3 + 1], bz = pos[ic * 3 + 2] - pos[id * 3 + 2];
      var nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx, l = Math.hypot(nx, ny, nz) || 1;
      nor[i0 * 3] = nx / l; nor[i0 * 3 + 1] = ny / l; nor[i0 * 3 + 2] = nz / l;
    }
    return spine;
  }
  function drawGL(turned) {
    var spine = bendLeaf(turned);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.p); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(buf.locs.aP); gl.vertexAttribPointer(buf.locs.aP, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.n); gl.bufferData(gl.ARRAY_BUFFER, nor, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(buf.locs.aN); gl.vertexAttribPointer(buf.locs.aN, 3, gl.FLOAT, false, 0, 0);
    gl.uniform2f(uni.uO, VW / 2, VH * 0.42); gl.uniform2f(uni.uH, VW / 2, VH / 2);
    gl.uniform1f(uni.uD, Math.max(3000, PW * 5)); gl.uniform1f(uni.uLift, Math.sin(turned * Math.PI));
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, anim.tf); gl.uniform1i(uni.uF, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, anim.tb); gl.uniform1i(uni.uB, 1);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibuf); gl.drawElements(gl.TRIANGLES, nIdx, gl.UNSIGNED_SHORT, 0);
    return spine;
  }
  // the clip-space centre used above (VH*0.42) is the perspective origin; pages at z=0 still land exactly on their DOM position
  // because w = 1 there, so the vertical origin only changes how the lifted paper leans.

  // ---- turn lifecycle
  function beginTurn(leaf, forward) {
    var front = PG[2 * leaf], back = PG[2 * leaf + 1];
    if (leaf === 0 && window.__heroHold) window.__heroHold(true);
    anim = { leaf: leaf, forward: forward, p: 0, front: front, back: back, ready: false, dead: false, t0: performance.now() };
    var a = anim;
    if (!initGL()) { a.ready = true; a.flat = true; showUnder(a); return; }
    Promise.all([ready(2 * leaf), ready(2 * leaf + 1)]).then(function (cvs) {
      if (a.dead || anim !== a) return;
      try {
      a.tf = texFor(2 * leaf); a.tb = texFor(2 * leaf + 1);
      var r = pages.getBoundingClientRect();
      geo.spine0 = r.left + PW; geo.top = r.top; geo.v0 = curV;
      sizeGL(); glc.style.display = 'block';
      drawGL(forward ? a.p : 1 - a.p);               // draw first, then hide the real pages in the same frame
      showUnder(a);
      a.ready = true; renderTurn(a.p); if (a.after) a.after();
      if (PERF) PERF.wait(performance.now() - a.t0);
      if (window.__dbg) console.log('turn ready after', Math.round(performance.now() - a.t0), 'ms', shots.map(function (x) { return x.dirty ? 'D' : (x.cv ? '.' : '-'); }).join(''));
      } catch (err) { if (window.__dbg) console.log('glfail', err && err.stack); a.ready = true; a.flat = true; showUnder(a); if (a.after) a.after(); }
    }, function (err) {
      if (window.__dbg) console.log('snapfail', err && err.message);
      if (a.dead || anim !== a) return;
      a.ready = true; a.flat = true; showUnder(a); if (a.after) a.after();
    });
  }
  function showUnder(a) {
    // the real pages beneath the moving leaf: both neighbours of the leaf are shown, the leaf's own two pages are hidden
    var sp = a.forward ? a.leaf : a.leaf + 1;
    hideAll(); shown = [];
    var L = a.forward ? leftOf(sp) : leftOf(sp - 1), R = a.forward ? rightOf(sp + 1) : rightOf(sp);
    setPage(L, 'L'); setPage(R, 'R'); shown = [L, R];
    if (a.flat) { setPage(a.forward ? a.front : a.back, a.forward ? 'R' : 'L'); }
  }
  function renderTurn(p) {
    var a = anim; if (!a) return; a.p = p;
    var turned = a.forward ? p : 1 - p;
    setV(a.leaf + turned);
    if (!a.ready) return;
    if (a.flat) {                                         // no WebGL: simple hinge on the real page
      var pg = turned < 0.5 ? a.front : a.back;
      setPage(pg, turned < 0.5 ? 'R' : 'L'); hideOther(pg === a.front ? a.back : a.front);
      pg.style.transformOrigin = turned < 0.5 ? '0 50%' : '100% 50%';
      pg.style.transform = 'perspective(2400px) rotateY(' + (turned < 0.5 ? -turned * 180 : (1 - turned) * 180) + 'deg)';
    } else drawGL(turned);
    var strength = Math.sin(turned * Math.PI), reach = Math.abs(Math.cos(turned * Math.PI)), side = turned < 0.5 ? 'R' : 'L';
    if (cast._side !== side) {                            // the shadow the lifted leaf throws on the page below; only transform and opacity change per frame
      cast._side = side; cast.style.width = (PW + 60) + 'px';
      cast.style.left = side === 'R' ? '50%' : 'calc(50% - ' + (PW + 60) + 'px)';
      cast.style.transformOrigin = side === 'R' ? '0 50%' : '100% 50%';
      cast.style.background = side === 'R' ? 'linear-gradient(90deg,rgba(40,20,0,0),rgba(40,20,0,.32) 75%,rgba(40,20,0,0))' : 'linear-gradient(90deg,rgba(40,20,0,0),rgba(40,20,0,.32) 25%,rgba(40,20,0,0))';
    }
    cast.style.transform = 'scaleX(' + ((reach * PW + 60) / (PW + 60)).toFixed(4) + ')';
    cast.style.opacity = (strength * 0.85).toFixed(3);
  }
  function hideOther(p) { if (p) { p.classList.add('off'); } }
  function endTurn(commit) {
    var a = anim; if (!a) return;
    a.dead = true; cast.style.opacity = 0; cast._side = null;
    if (commit) s += a.forward ? 1 : -1;
    a.front.style.transform = a.back.style.transform = '';
    showSpread(s); setV(s); anim = null;
    if (glc) { requestAnimationFrame(function () { if (!anim && glc) glc.style.display = 'none'; }); }   // the real pages are painted before the picture goes
    opened(s); setTimeout(pumpShots, 1400);           // let the page's own reveal animations finish first
  }

  // ---------------------------------------------------------------- animation driver
  var raf = 0;
  function runTo(goal, dur, done) {
    var from = anim.p, t0 = performance.now(), a = anim;
    if (reduceMotion) dur = 40;
    cancelAnimationFrame(raf); if (PERF) PERF.start();
    (function tick(now) {
      var t = Math.min(1, (now - t0) / dur), e = t * t * (3 - 2 * t);
      renderTurn(from + (goal - from) * e);
      if (t < 1) raf = requestAnimationFrame(tick); else { if (PERF) PERF.end(); endTurn(goal === 1); if (done) done(); pump(); }
    })(t0);
  }
  function turn(forward, dur) {
    if (anim) return;
    var leaf = forward ? s : s - 1; if (leaf < 0 || leaf > 5) return;
    beginTurn(leaf, forward);
    var go = function () { if (anim) runTo(1, dur || 780); };
    if (anim.ready) go(); else anim.after = go;
  }
  function pump() { if (anim) return; if (target > s) turn(true, Math.abs(target - s) > 1 ? 380 : 780); else if (target < s) turn(false, Math.abs(target - s) > 1 ? 380 : 780); }
  function goTo(n) { target = clamp(n, 0, 6); pump(); }

  // ---------------------------------------------------------------- drag / swipe / keys / tabs
  var drag = null, justDragged = false, NODRAG = 'textarea,input,select,.tbox,.tw,.cvs';
  bk.addEventListener('pointerdown', function (e) {
    if (anim || e.button > 0 || (e.target.closest && e.target.closest(NODRAG))) { drag = null; return; }
    if (e.target.closest && e.target.closest('.bt2')) { drag = null; return; }
    drag = { x: e.clientX, y: e.clientY, active: false, id: e.pointerId };
    if (s === 0 && window.__heroHold) window.__heroHold(true);
    warm();
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
    if (!d.active) { if (s === 0 && !anim && window.__heroHold) window.__heroHold(false); return; }
    justDragged = true; setTimeout(function () { justDragged = false; }, 60);
    var p = anim.p, ok = p > 0.22, dur = 780 * (0.4 + 0.6 * Math.abs((ok ? 1 : 0) - p));
    if (!ok) target = s;
    var go = function () { runTo(ok ? 1 : 0, dur); };
    if (anim.ready) go(); else anim.after = go;
  }
  bk.addEventListener('pointerenter', function () { if (s === 0 && !anim && window.__heroHold) { window.__heroHold(true); } warm(); });
  bk.addEventListener('pointerleave', function () { if (s === 0 && !anim && !drag && window.__heroHold) window.__heroHold(false); });
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
  window.addEventListener('resize', function () { size(); if (anim) { endTurn(false); } setV(s); shots.forEach(function (x) { x.dirty = true; }); pumpShots(); });
  window.addEventListener('hashchange', function () { var i = SLUG.indexOf(location.hash.slice(1)); if (i >= 0) goTo(i); });


  // ---------------------------------------------------------------- ?perf: an on-screen meter (only when the address has ?perf)
  var PERF = null;
  if (/[?&]perf\b/.test(location.search)) (function () {
    var box = el('div', ''); box.style.cssText = 'position:fixed;left:8px;top:8px;z-index:99;background:rgba(0,0,0,.82);color:#9f9;font:12px/1.45 monospace;padding:8px 10px;border-radius:6px;white-space:pre;pointer-events:none;max-width:46vw';
    document.body.appendChild(box);
    var snaps = [], waits = [], turns = [], gaps = [], cur = null, lastT = 0, ua = navigator.userAgent;
    var eng = /Firefox/.test(ua) ? 'Firefox' : (/Chrome|Chromium|Edg/.test(ua) ? 'Chromium' : (/Safari/.test(ua) ? 'Safari' : '?'));
    (function loop(t) { if (lastT && t - lastT > 50) gaps.push([Math.round(t - lastT), anim ? 'turn' : 'rest']); lastT = t; if (cur) cur.push(t); requestAnimationFrame(loop); })(0);
    function st(a) { if (!a.length) return '-'; var b = a.slice().sort(function (x, y) { return x - y; }); return 'n' + b.length + ' med ' + Math.round(b[b.length >> 1]) + ' max ' + Math.round(b[b.length - 1]); }
    PERF = {
      snap: function (i, ms) { snaps.push(ms); },
      wait: function (ms) { waits.push(ms); },
      start: function () { cur = []; },
      end: function () { if (!cur) return; var d = []; for (var k = 1; k < cur.length; k++) d.push(cur[k] - cur[k - 1]); cur = null; if (!d.length) return; d.sort(function (x, y) { return x - y; }); turns.push({ n: d.length, med: d[d.length >> 1], p95: d[Math.floor(d.length * 0.95)], max: d[d.length - 1], slow: d.filter(function (x) { return x > 20; }).length }); }
    };
    setInterval(function () {
      var gl2 = null; try { var c = document.createElement('canvas').getContext('webgl'), x = c && c.getExtension('WEBGL_debug_renderer_info'); gl2 = x ? c.getParameter(x.UNMASKED_RENDERER_WEBGL) : (c ? 'webgl' : 'none'); } catch (e) {}
      box.textContent = 'engine ' + eng + '  dpr ' + (window.devicePixelRatio || 1) + '  ' + innerWidth + 'x' + innerHeight + '  leaf ' + (gl === false ? 'flat' : (gl ? 'webgl' : 'not yet')) +
        '\npage pictures ms: ' + st(snaps) + '\nwait before turn ms: ' + st(waits) +
        '\nturns (frame ms): ' + turns.slice(-6).map(function (t) { return t.n + 'f med' + t.med.toFixed(0) + ' p95 ' + t.p95.toFixed(0) + ' max ' + t.max.toFixed(0) + ' slow' + t.slow; }).join('\n   ') +
        '\nfreezes >50ms: ' + gaps.length + '  last: ' + gaps.slice(-8).map(function (g) { return g[0] + (g[1] === 'turn' ? 't' : ''); }).join(' ');
      PERF.gl = gl2;
    }, 500);
  })();

  // ---------------------------------------------------------------- start
  var start = Math.max(0, SLUG.indexOf(location.hash.slice(1)));
  s = target = start; showSpread(s); setV(s); opened(s);
  // photograph the pages once the fonts are in, nearest pages first
  (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(function () { setTimeout(pumpShots, 400); });
  try { reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e) {}
})();
