/*
 * TalkingPortrait — Anvaya AI Lab
 * Animates a single portrait in the browser (WebGL mesh warp):
 *   - lip sync driven by the agent's real audio (Web Audio analyser)
 *   - natural blinks, eye saccades, breathing and head sway
 *   - attentive look + small nods while the user is speaking
 * No server GPU, no per-minute cost. Rig comes from tools/build_face_rig.py.
 */
(function (global) {
  "use strict";

  const VS = `
    attribute vec2 a_pos; attribute vec2 a_uv; attribute float a_type; attribute float a_mv;
    uniform vec4 u_view;   // scaleX, scaleY, offsetX, offsetY (image px -> clip)
    varying vec2 v_uv; varying float v_type; varying float v_mv;
    void main() {
      vec2 p = a_pos * u_view.xy + u_view.zw;
      gl_Position = vec4(p.x, -p.y, 0.0, 1.0);
      v_uv = a_uv; v_type = a_type; v_mv = a_mv;
    }`;
  const FS = `
    precision mediump float;
    uniform sampler2D u_tex; uniform float u_open;
    varying vec2 v_uv; varying float v_type; varying float v_mv;
    void main() {
      vec4 c = texture2D(u_tex, v_uv);
      if (v_type > 0.5 && v_type < 1.5) {           // mouth interior
        vec3 teeth = vec3(0.66, 0.60, 0.55);
        vec3 dark  = vec3(0.13, 0.05, 0.04);
        float t = smoothstep(0.02, 0.24, v_mv);         // thin, soft upper-teeth band
        vec3 m = mix(teeth, dark, t);
        m = mix(m, vec3(0.36, 0.13, 0.11), smoothstep(0.70, 1.0, v_mv) * 0.55); // tongue/lip shadow
        c = vec4(mix(c.rgb, m, clamp(u_open * 6.0, 0.0, 1.0)), 1.0);
      }
      // feather the left/right picture edges into the blurred page background
      c.a *= smoothstep(0.0, 0.05, v_uv.x) * smoothstep(1.0, 0.95, v_uv.x);
      gl_FragColor = c;
    }`;

  const rand = (a, b) => a + Math.random() * (b - a);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

  class TalkingPortrait {
    constructor(canvas, opts) {
      this.canvas = canvas;
      this.imgUrl = opts.image;
      this.rigUrl = opts.rig;
      this.showFrac = opts.showFrac ?? 0.80;   // fraction of the image height that fills the canvas
      this.state = {
        open: 0, round: 0, blink: 0, gx: 0, gy: 0, tgx: 0, tgy: 0,
        nod: 0, tilt: 0, level: -100, peak: -40, listening: false, speaking: false,
      };
      this.nextBlink = performance.now() + rand(1500, 4000);
      this.nextSaccade = performance.now() + rand(800, 2000);
      this.analyser = null;
      this.running = false;
    }

    async init() {
      const [img, rig] = await Promise.all([
        new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = this.imgUrl; }),
        fetch(this.rigUrl).then(r => { if (!r.ok) throw new Error("rig " + r.status); return r.json(); }),
      ]);
      this.img = img; this.rig = rig;
      const gl = this.canvas.getContext("webgl", { premultipliedAlpha: false, alpha: true, antialias: true, preserveDrawingBuffer: true });
      if (!gl) throw new Error("WebGL not available");
      this.gl = gl;
      const prog = this._program(VS, FS);
      this.prog = prog;
      gl.useProgram(prog);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      this.loc = {
        pos: gl.getAttribLocation(prog, "a_pos"), uv: gl.getAttribLocation(prog, "a_uv"),
        type: gl.getAttribLocation(prog, "a_type"), mv: gl.getAttribLocation(prog, "a_mv"),
        view: gl.getUniformLocation(prog, "u_view"), open: gl.getUniformLocation(prog, "u_open"),
      };
      // texture
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

      // static expanded buffers (uv, type, mv); positions updated per frame
      const T = rig.tris, n = T.length * 3;
      this.idx = new Uint16Array(n);
      const uv = new Float32Array(n * 2), type = new Float32Array(n), mv = new Float32Array(n);
      T.forEach((t, k) => {
        for (let j = 0; j < 3; j++) {
          const v = t[j], o = k * 3 + j;
          this.idx[o] = v;
          uv[o * 2] = rig.verts[v][0] / rig.w; uv[o * 2 + 1] = rig.verts[v][1] / rig.h;
          type[o] = t[3]; mv[o] = rig.mv[v] || 0;
        }
      });
      this.posBuf = gl.createBuffer();
      this.pos = new Float32Array(n * 2);
      this.cur = rig.verts.map(p => [p[0], p[1]]);
      const bind = (data, loc, size) => {
        const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      };
      bind(uv, this.loc.uv, 2); bind(type, this.loc.type, 1); bind(mv, this.loc.mv, 1);
      this.count = n;
      this._resize();
      this._ro = new ResizeObserver(() => this._resize());
      this._ro.observe(this.canvas);
      return this;
    }

    _program(vs, fs) {
      const gl = this.gl;
      const sh = (type, src) => {
        const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        return s;
      };
      const p = gl.createProgram();
      gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      return p;
    }

    _resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const r = this.canvas.getBoundingClientRect();
      const cw = Math.max(1, Math.round(r.width * dpr)), ch = Math.max(1, Math.round(r.height * dpr));
      if (this.canvas.width !== cw || this.canvas.height !== ch) { this.canvas.width = cw; this.canvas.height = ch; }
      this.gl.viewport(0, 0, cw, ch);
      // Framing: show the whole head (top of image) down to the upper chest, centred on the
      // face. Areas outside the picture stay transparent; the page shows a blurred copy there.
      const { w, h, face } = this.rig;
      const s = ch / (h * this.showFrac);
      let ox = cw / 2 - face.cx * s, oy = 0;
      if (w * s >= cw) ox = clamp(ox, cw - w * s, 0);
      this.offsetPx = [ox, oy]; this.scale = s;
      this.view = [2 * s / cw, 2 * s / ch, 2 * ox / cw - 1, 2 * oy / ch - 1];
    }

    // ---- inputs -------------------------------------------------------------
    attachAudio(mediaStreamTrack, audioCtx) {
      this.detachAudio();
      if (!mediaStreamTrack) return;
      this.audioCtx = audioCtx || this.audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      const src = this.audioCtx.createMediaStreamSource(new MediaStream([mediaStreamTrack]));
      const an = this.audioCtx.createAnalyser();
      an.fftSize = 1024; an.smoothingTimeConstant = 0.35;
      src.connect(an);
      this._src = src; this.analyser = an;
      this.freq = new Float32Array(an.frequencyBinCount);
    }
    detachAudio() {
      try { this._src && this._src.disconnect(); } catch (_) {}
      this._src = null; this.analyser = null;
    }
    setListening(on) {
      const st = this.state;
      if (st.listening && !on) this._nodAt = performance.now();   // acknowledge when user finishes
      st.listening = on;
    }

    start() { if (!this.running) { this.running = true; this._t0 = performance.now(); requestAnimationFrame(t => this._frame(t)); } }
    stop() { this.running = false; }

    // ---- per-frame ------------------------------------------------------------
    _audioFeatures() {
      if (!this.analyser) return { open: 0, round: 0 };
      const an = this.analyser, f = this.freq;
      an.getFloatFrequencyData(f);
      const hz = this.audioCtx.sampleRate / an.fftSize;
      let low = 0, mid = 0, high = 0;
      for (let i = 1; i < f.length; i++) {
        const fr = i * hz, p = Math.pow(10, f[i] / 10);
        if (fr < 150) continue;
        if (fr < 900) low += p; else if (fr < 2500) mid += p; else if (fr < 5000) high += p; else break;
      }
      const tot = low + mid + high;
      const lvl = 10 * Math.log10(tot + 1e-12);
      const st = this.state;
      st.peak = Math.max(lvl, st.peak - 0.05);                // slow-decay peak = auto gain
      const open = Math.pow(clamp((lvl - (st.peak - 30)) / 30, 0, 1), 1.4) * 0.9;
      const round = tot > 0 ? clamp((low / tot - 0.55) / 0.3, 0, 1) : 0;
      return { open: lvl < -85 ? 0 : open, round };
    }

    _frame(now) {
      if (!this.running) return;
      const st = this.state, rig = this.rig, F = rig.face;
      const t = (now - this._t0) / 1000;
      const dt = Math.min(0.05, (now - (this._last || now)) / 1000) || 0.016;
      this._last = now;

      // mouth from audio (fast attack, slower release)
      const a = this._audioFeatures();
      const k = a.open > st.open ? 0.55 : 0.28;
      st.open += (a.open - st.open) * k;
      st.round += (a.round * (a.open > 0.15 ? 1 : 0) - st.round) * 0.2;
      st.speaking = st.open > 0.08;

      // blinks
      if (now > this.nextBlink && !this._blinkStart) this._blinkStart = now;
      if (this._blinkStart) {
        const e = (now - this._blinkStart) / 1000;
        st.blink = e < 0.07 ? e / 0.07 : e < 0.17 ? 1 - (e - 0.07) / 0.1 : 0;
        if (e >= 0.17) {
          this._blinkStart = null; st.blink = 0;
          this.nextBlink = now + (Math.random() < 0.15 ? 220 : rand(2200, 5500));   // sometimes a double blink
        }
      }

      // gaze: look at the user while they talk, wander a little otherwise
      if (now > this.nextSaccade) {
        if (st.listening) { st.tgx = rand(-0.1, 0.1); st.tgy = rand(-0.05, 0.05); }
        else if (st.speaking) { st.tgx = rand(-0.35, 0.35); st.tgy = rand(-0.2, 0.1); }
        else { st.tgx = rand(-0.5, 0.5); st.tgy = rand(-0.25, 0.15); }
        if (Math.random() < 0.35) { st.tgx = 0; st.tgy = 0; }
        this.nextSaccade = now + (st.listening ? rand(1500, 3000) : rand(700, 2200));
      }
      st.gx += (st.tgx - st.gx) * 0.35; st.gy += (st.tgy - st.gy) * 0.35;

      // head: breathing + sway, livelier when speaking, attentive tilt when listening
      const sp = st.speaking ? 1 : 0;
      let theta = 0.010 * Math.sin(t * 0.55) + 0.004 * Math.sin(t * 1.3 + 1) + sp * 0.012 * Math.sin(t * 2.3);
      let tx = F.fw * (0.004 * Math.sin(t * 0.4 + 2) + sp * 0.003 * Math.sin(t * 1.9));
      let ty = F.fh * (0.004 * Math.sin(t * 1.1) + sp * 0.006 * st.open);
      st.tilt += ((st.listening ? 0.025 : 0) - st.tilt) * 0.05;
      theta += st.tilt;
      if (this._nodAt) {
        const e = (now - this._nodAt) / 1000;
        st.nod = e < 0.6 ? Math.sin((e / 0.6) * Math.PI) : 0;
        if (e >= 0.6) this._nodAt = null;
      }
      ty += st.nod * F.fh * 0.025;

      // deform
      const V = rig.verts, cur = this.cur, n = V.length;
      const jawD = st.open * 0.085 * F.fh * (1 - 0.35 * st.round);
      const rnd = st.round * st.open * 0.30 * F.mouthHW;
      const px = F.cx, py = F.chinY + 0.15 * F.fh;
      const c = Math.cos(theta), s = Math.sin(theta);
      const ex = { [rig.eyes.r.v]: rig.eyes.r.r, [rig.eyes.l.v]: rig.eyes.l.r };
      for (let i = 0; i < n; i++) {
        let x = V[i][0], y = V[i][1];
        y += rig.jaw[i] * jawD - rig.up[i] * jawD;
        x += rig.round[i] * rnd;
        y += rig.blink[i] * st.blink;
        if (ex[i] !== undefined) { x += st.gx * ex[i] * 0.55; y += st.gy * ex[i] * 0.40; }
        const hw = rig.head[i];
        if (hw > 0) {
          const dx = x - px, dy = y - py;
          const qx = px + dx * c - dy * s + tx, qy = py + dx * s + dy * c + ty;
          x += (qx - x) * hw; y += (qy - y) * hw;
        }
        cur[i][0] = x; cur[i][1] = y;
      }
      const P = this.pos, idx = this.idx;
      for (let o = 0; o < idx.length; o++) { const v = cur[idx[o]]; P[o * 2] = v[0]; P[o * 2 + 1] = v[1]; }

      const gl = this.gl;
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.prog);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
      gl.bufferData(gl.ARRAY_BUFFER, P, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.loc.pos); gl.vertexAttribPointer(this.loc.pos, 2, gl.FLOAT, false, 0, 0);
      gl.uniform4fv(this.loc.view, this.view);
      gl.uniform1f(this.loc.open, st.open);
      gl.drawArrays(gl.TRIANGLES, 0, this.count);
      requestAnimationFrame(tt => this._frame(tt));
    }
  }

  global.TalkingPortrait = TalkingPortrait;
})(window);
