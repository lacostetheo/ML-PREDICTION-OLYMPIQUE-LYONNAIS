/* OL·LAB — front end. No ML runs here: every number comes from the Python pipeline (site/data.js). */
(() => {
  "use strict";
  const D = window.OLDATA;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const S = { league: "F1", team: "Lyon", tab: "lab", opp: null, venue: "home", custom: {} };
  const C = { cyan: "#35e0ff", magenta: "#ff3cac", amber: "#ffb347", green: "#3dffa0", violet: "#a78bfa", grey: "#8c99c4", white: "#e8eeff" };

  const pct = (x) => Math.round(x * 100) + "%";
  const pct1 = (x) => (x * 100).toFixed(1) + "%";
  const range = (ci) => `${Math.round(ci[0] * 100)}–${Math.round(ci[1] * 100)}%`;
  const ord = (n) => n + (["th", "st", "nd", "rd"][(n % 100 > 10 && n % 100 < 14) ? 0 : (n % 10 < 4 ? n % 10 : 0)]);
  const fmtDate = (d, t) => new Date(d + "T12:00:00").toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" }) + (t ? " · " + t : "");
  const teams = () => D.leagues[S.league].teams;
  const info = () => D.leagues[S.league].info[S.team];

  /* arr layout of a pair: [pA,pD,pH, A_lo,A_hi, D_lo,D_hi, H_lo,H_hi, contrib x8, eloDiff] */
  function persp(home, team, arr) {
    const h = home === team;
    return { win: h ? arr[2] : arr[0], draw: arr[1], loss: h ? arr[0] : arr[2],
      winCI: h ? [arr[7], arr[8]] : [arr[3], arr[4]], drawCI: [arr[5], arr[6]], lossCI: h ? [arr[3], arr[4]] : [arr[7], arr[8]],
      ep: 3 * (h ? arr[2] : arr[0]) + arr[1] };
  }
  function pbar(p, big, labels) {
    const seg = (cls, v, t) => `<div class="seg ${cls}" style="--w:${(v * 100).toFixed(2)}%" title="${t} ${pct1(v)}">${v > 0.07 ? pct(v) : ""}</div>`;
    return `<div class="pbar ${big ? "big" : ""}">${seg("win", p.win, "win")}${seg("draw", p.draw, "draw")}${seg("loss", p.loss, "loss")}</div>` +
      (labels ? `<div class="plegend"><span><i style="background:${C.cyan}"></i>${labels[0]} wins</span><span><i style="background:${C.grey}"></i>Draw</span><span><i style="background:${C.magenta}"></i>${labels[1]} wins</span></div>` : "");
  }
  function lightUp(root) { requestAnimationFrame(() => requestAnimationFrame(() => $$(".pbar", root).forEach((b) => b.classList.add("on")))); if (REDUCED) $$(".pbar", root).forEach((b) => b.classList.add("on")); }

  /* ------------------------------------------------------------------ pickers */
  function fillLeagues() {
    $("#league").innerHTML = Object.entries(D.leagues).map(([k, v]) => `<option value="${k}">${v.name}</option>`).join("");
    $("#league").value = S.league;
  }
  function fillTeams() {
    $("#team").innerHTML = teams().map((t) => `<option>${t}</option>`).join("");
    $("#team").value = S.team;
  }
  function resetMatchDefaults() {
    const up = S.league === "F1" ? (D.calendar_f1[S.team] || [])[0] : null, nx = up || D.next[S.league][S.team];
    S.opp = nx.home === S.team ? nx.away : nx.home;
    S.venue = nx.home === S.team ? "home" : "away";
  }

  /* ------------------------------------------------------------------ MATCH LAB */
  function renderTeamCard() {
    const i = info(), n = teams().length;
    const sv = i.sv === null ? "n/a" : "×" + Math.exp(i.sv).toFixed(2);
    $("#teamcard").innerHTML = `<div class="teamcard"><div class="name">${S.team}</div><div class="sub">${D.leagues[S.league].name} · snapshot ${D.as_of}</div>
      <div class="stats">
        <div class="stat"><b>${Math.round(i.elo)}</b><span>Elo · ${ord(i.elo_rank)}/${n}</span></div>
        <div class="stat"><b>${sv}</b><span>Squad value vs league avg</span></div>
        <div class="stat"><b>${i.pts10.toFixed(2)}</b><span>Points / match (last 10)</span></div>
        <div class="stat"><b>${i.gf10.toFixed(1)}–${i.ga10.toFixed(1)}</b><span>Goals for–against / match</span></div>
      </div><canvas id="spark" height="90" aria-label="Elo rating history"></canvas><div class="note">Elo rating since 2010 (monthly)</div></div>`;
    drawSpark(i.elo_series);
  }
  function drawSpark(series) {
    const cv = $("#spark"); if (!cv) return;
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = 90;
    cv.width = w * dpr; cv.height = h * dpr; const g = cv.getContext("2d"); g.scale(dpr, dpr);
    const vals = series.filter((v) => v !== null); if (!vals.length) return;
    const lo = Math.min(...vals) - 20, hi = Math.max(...vals) + 20;
    const X = (k) => 6 + (k / (series.length - 1)) * (w - 12), Y = (v) => h - 8 - ((v - lo) / (hi - lo)) * (h - 20);
    g.strokeStyle = "rgba(53,224,255,.12)"; g.lineWidth = 1;
    [lo + 20, (lo + hi) / 2, hi - 20].forEach((v) => { g.beginPath(); g.moveTo(0, Y(v)); g.lineTo(w, Y(v)); g.stroke(); });
    const grad = g.createLinearGradient(0, 0, w, 0); grad.addColorStop(0, C.cyan); grad.addColorStop(1, C.magenta);
    g.strokeStyle = grad; g.lineWidth = 2; g.shadowColor = C.cyan; g.shadowBlur = 8; g.beginPath(); let pen = false;
    series.forEach((v, k) => { if (v === null) { pen = false; return; } pen ? g.lineTo(X(k), Y(v)) : g.moveTo(X(k), Y(v)); pen = true; });
    g.stroke();
    const last = series.length - 1 - [...series].reverse().findIndex((v) => v !== null);
    g.fillStyle = C.magenta; g.shadowColor = C.magenta; g.beginPath(); g.arc(X(last), Y(series[last]), 4, 0, 7); g.fill();
  }

  function matchCard(n, home, away, date, time, arr, market, extra, stale) {
    const t = S.team, h = home === t, opp = h ? away : home, p = persp(home, t, arr);
    const mk = market ? persp(home, t, [market[0], market[1], market[2], 0, 0, 0, 0, 0, 0]) : null;
    return `<article class="mcard ${n === 1 ? "first" : ""}"><div class="meta"><span class="badge-n" ${stale ? 'style="background:rgba(255,179,71,.15);color:#ffb347"' : ""}>${stale ? "ROUND PLAYED" : n === 1 ? "NEXT" : "THEN"} · #${n}</span><span>${fmtDate(date, time)}${extra || ""}</span></div>
      <h4>${t}<span class="venue ${h ? "H" : "A"}">${h ? "H" : "A"}</span> <span class="vs">vs</span> ${opp}</h4>
      ${pbar(p, false)}
      <div class="mline"><span>${t} win <b>${pct(p.win)}</b> <small>(90 %: ${range(p.winCI)})</small></span><span>draw <b>${pct(p.draw)}</b></span><span>${opp} <b>${pct(p.loss)}</b></span></div>
      <div class="mline"><span>expected points <b>${p.ep.toFixed(2)}</b></span>${mk ? `<span>bookmaker: ${pct(mk.win)} / ${pct(mk.draw)} / ${pct(mk.loss)}</span>` : ""}</div></article>`;
  }
  function customCard(idx) {
    const key = S.league + "|" + S.team, c = (S.custom[key] = S.custom[key] || [{}, {}])[idx - 2];
    const opts = teams().filter((x) => x !== S.team);
    const head = `<article class="mcard"><div class="meta"><span class="badge-n">THEN · #${idx}</span><span>calendar not available</span></div>`;
    const sel = `<div class="pick"><select data-c="${idx - 2}" class="c-opp" aria-label="Opponent for match ${idx}"><option value="">— choose opponent —</option>${opts.map((o) => `<option ${c.opp === o ? "selected" : ""}>${o}</option>`).join("")}</select>
      <div class="seg-toggle"><button class="c-v ${c.venue !== "away" ? "on" : ""}" data-c="${idx - 2}" data-v="home">H</button><button class="c-v ${c.venue === "away" ? "on" : ""}" data-c="${idx - 2}" data-v="away">A</button></div></div>`;
    if (!c.opp) return head + `<div class="empty">Only the next round is published for this league in the data source. Choose an opponent to see the forecast.</div>` + sel + `</article>`;
    const home = c.venue === "away" ? c.opp : S.team, away = c.venue === "away" ? S.team : c.opp;
    const arr = D.pairs[S.league][`${home}|${away}`], p = persp(home, S.team, arr);
    return head + sel + pbar(p, false) + `<div class="mline"><span>${S.team} win <b>${pct(p.win)}</b> <small>(90 %: ${range(p.winCI)})</small></span><span>draw <b>${pct(p.draw)}</b></span><span>${c.opp} <b>${pct(p.loss)}</b></span></div>
      <div class="mline"><span>expected points <b>${p.ep.toFixed(2)}</b></span><span>neutral schedule</span></div></article>`;
  }
  function renderNext3() {
    const nx = D.next[S.league][S.team];
    $("#next-team").textContent = S.team;
    let html = "";
    if (S.league === "F1") {
      (D.calendar_f1[S.team] || []).forEach((m, k) => {
        const real = nx.home === m.home && nx.away === m.away && nx.date === m.date;
        html += real ? matchCard(k + 1, nx.home, nx.away, nx.date, nx.time, nx.p.concat(nx.ci), nx.market, ' · <b style="color:#3dffa0">confirmed</b>')
                     : matchCard(k + 1, m.home, m.away, m.date, "", D.pairs.F1[`${m.home}|${m.away}`], null, ` · matchday ${m.matchday}`);
      });
      $("#next3-note").textContent = "From today (" + D.as_of + ") on, per the published Ligue 1 calendar (dates can change). Only a match listed by the odds source has bookmaker prices and real rest days; later matches use a neutral schedule. The further ahead, the less today's snapshot reflects reality — Elo and form move after every match.";
    } else {
      const stale = nx.date < D.as_of;
      html = matchCard(1, nx.home, nx.away, nx.date, nx.time, nx.p.concat(nx.ci), nx.market, stale ? " · played, result not in the data yet" : ' · <b style="color:#3dffa0">confirmed</b>', stale) + customCard(2) + customCard(3);
      $("#next3-note").textContent = "Match 1 is the fixture listed by the odds source" + (stale ? " (already played before the snapshot: this is the forecast the model would have made). " : ". ") + "For this league the calendar beyond that round is not part of the data source, so matches 2–3 are yours to choose.";
    }
    $("#next3").innerHTML = html; lightUp($("#next3"));
    $$(".c-opp").forEach((s) => s.addEventListener("change", (e) => { const key = S.league + "|" + S.team; S.custom[key][+e.target.dataset.c].opp = e.target.value || undefined; renderNext3(); }));
    $$(".c-v").forEach((b) => b.addEventListener("click", (e) => { const key = S.league + "|" + S.team; S.custom[key][+b.dataset.c].venue = b.dataset.v; renderNext3(); }));
  }

  function renderSimulator() {
    const opts = teams().filter((x) => x !== S.team);
    $("#opp").innerHTML = opts.map((o) => `<option>${o}</option>`).join(""); $("#opp").value = S.opp;
    $("#v-home").classList.toggle("on", S.venue === "home"); $("#v-away").classList.toggle("on", S.venue === "away");
    const home = S.venue === "home" ? S.team : S.opp, away = S.venue === "home" ? S.opp : S.team;
    const arr = D.pairs[S.league][`${home}|${away}`], p = persp(home, S.team, arr);
    const eloDiff = (S.venue === "home" ? 1 : -1) * arr[17];
    $("#match-title").innerHTML = `${home} <small>vs</small> ${away}<br><small>${S.team} playing ${S.venue} · neutral schedule (7 days' rest each)</small>`;
    $("#match-bar").innerHTML = pbar(p, true, [S.team, S.opp]); lightUp($("#match-bar"));
    const nx = D.next[S.league][S.team]; const isNext = nx.home === home && nx.away === away;
    const mk = isNext ? persp(home, S.team, [nx.market[0], nx.market[1], nx.market[2], 0, 0, 0, 0, 0, 0]) : null;
    $("#match-facts").innerHTML = `<div class="stat"><b>${p.ep.toFixed(2)}</b><span>${S.team} expected points</span></div>
      <div class="stat"><b>${range(p.winCI)}</b><span>90 % interval, ${S.team} win</span></div>
      <div class="stat"><b>${eloDiff > 0 ? "+" : ""}${Math.round(eloDiff)}</b><span>Elo edge for ${S.team}</span></div>` +
      (mk ? `<div class="stat" style="grid-column:1/-1"><b>${pct(mk.win)} / ${pct(mk.draw)} / ${pct(mk.loss)}</b><span>bookmaker's view of this real fixture (margin removed) — in testing the bookmaker beat our model</span></div>` : "");
    renderWaterfall(arr);
  }
  function renderWaterfall(arr) {
    const sign = S.venue === "home" ? 1 : -1;
    const items = D.families.map((f, i) => ({ f, v: sign * arr[9 + i] })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
    const total = items.reduce((s, x) => s + x.v, 0), max = Math.max(0.2, ...items.map((x) => Math.abs(x.v)), Math.abs(total));
    const row = (lab, v, cls) => `<div class="wf-row ${cls || ""}"><div class="lab">${lab}</div><div class="wf-track"><div class="wf-bar ${v >= 0 ? "pos" : "neg"}" data-w="${(Math.abs(v) / max * 49).toFixed(1)}"></div></div><div class="wf-val" style="color:${v >= 0 ? C.cyan : C.magenta}">${v >= 0 ? "+" : ""}${v.toFixed(2)}</div></div>`;
    $("#waterfall").innerHTML = `<div class="wf-axis"><span>← favours ${S.opp}</span><span>favours ${S.team} →</span></div>` + items.map((x) => row(x.f, x.v)).join("") +
      row("<b>Net push</b>", total, "wf-total");
    animateBars();
  }
  function animateBars() {
    $$(".wf-bar").forEach((b, i) => { b.style.width = "0"; setTimeout(() => (b.style.width = b.dataset.w + "%"), REDUCED ? 0 : 120 + i * 130); });
  }

  function renderLab() { renderTeamCard(); renderNext3(); renderSimulator(); $("#asof").textContent = `Team state as of ${D.as_of} · walk-forward logistic regression · all numbers precomputed by the Python pipeline`; $("#asof2").textContent = D.as_of; }

  /* ------------------------------------------------------------------ MONTE CARLO */
  const MC = { cache: {}, k: 0, playing: false, raf: 0, last: -1 };
  const SPEEDS = [1, 3, 10, 40, 150];
  function dec(b64) { const bin = atob(b64), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
  function mcData() {
    const L = S.league;
    if (!MC.cache[L]) { const s = D.leagues[L].sim; MC.cache[L] = { pos: dec(s.positions), pts: dec(s.points), T: s.teams.length, N: s.n_replay, teams: s.teams }; }
    return MC.cache[L];
  }
  function mcReset(draw = true) {
    cancelAnimationFrame(MC.raf); MC.playing = false; $("#mc-play").textContent = "▶ Play";
    const d = mcData(), ti = d.teams.indexOf(S.team);
    MC.k = 0; MC.last = -1; MC.counts = new Int32Array(d.T);
    let lo = 255, hi = 0; for (let n = 0; n < d.N; n++) { const v = d.pts[n * d.T + ti]; if (v < lo) lo = v; if (v > hi) hi = v; }
    MC.pmin = lo; MC.pc = new Int32Array(hi - lo + 1);
    $("#mc-total").textContent = d.N.toLocaleString("en-GB");
    $("#mc-team-lead").textContent = S.team + "'s";
    if (draw) mcDraw();
  }
  function mcStep(n) {
    const d = mcData(), ti = d.teams.indexOf(S.team);
    for (let i = 0; i < n && MC.k < d.N; i++, MC.k++) { MC.counts[d.pos[MC.k * d.T + ti] - 1]++; MC.pc[d.pts[MC.k * d.T + ti] - MC.pmin]++; MC.last = MC.k; }
  }
  function mcDraw() {
    const d = mcData(), ti = d.teams.indexOf(S.team), lg = D.leagues[S.league], T = d.T, k = Math.max(MC.k, 1);
    $("#mc-n").textContent = MC.k.toLocaleString("en-GB");
    const target = lg.sim.pos_probs[S.team];
    // --- position histogram (SVG)
    const W = 560, H = 270, ml = 38, mb = 30, mt = 10, bw = (W - ml - 8) / T;
    const freq = Array.from(MC.counts, (c) => c / k), ymax = Math.max(0.05, ...target, ...(MC.k > 30 ? freq : [0])) * 1.12;
    const Y = (v) => mt + (H - mt - mb) * (1 - v / ymax), cl = lg.slots.champions_league, eu = lg.slots.europe, rel = lg.slots.relegation;
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Distribution of final position">`;
    const band = (a, b, col, lab) => `<rect x="${ml + (a - 1) * bw}" y="${mt}" width="${(b - a + 1) * bw}" height="${H - mt - mb}" fill="${col}"/><text x="${ml + ((a - 1) + (b - a + 1) / 2) * bw}" y="${mt + 11}" fill="#93a1c4" font-size="9" text-anchor="middle">${lab}</text>`;
    svg += band(1, cl, "rgba(53,224,255,.07)", "Champions L.") + band(cl + 1, eu, "rgba(167,139,250,.07)", "Europe") + band(T - rel + 1, T, "rgba(255,60,172,.09)", "Relegation");
    for (let g = 0; g <= 4; g++) { const v = ymax * g / 4; svg += `<line x1="${ml}" x2="${W - 8}" y1="${Y(v)}" y2="${Y(v)}" stroke="#1f2c4a"/><text x="${ml - 6}" y="${Y(v) + 3}" fill="#93a1c4" font-size="10" text-anchor="end">${Math.round(v * 100)}%</text>`; }
    for (let i = 0; i < T; i++) {
      const x = ml + i * bw + 2, w = bw - 4;
      svg += `<rect x="${x}" y="${Y(freq[i])}" width="${w}" height="${H - mb - Y(freq[i])}" rx="3" fill="url(#gb)"/>`;
      svg += `<rect x="${x}" y="${Y(target[i])}" width="${w}" height="${H - mb - Y(target[i])}" rx="3" fill="none" stroke="${C.amber}" stroke-width="1.4" stroke-dasharray="3 3"/>`;
      svg += `<text x="${x + w / 2}" y="${H - 12}" fill="#93a1c4" font-size="10" text-anchor="middle">${i + 1}</text>`;
    }
    svg += `<defs><linearGradient id="gb" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${C.cyan}"/><stop offset="1" stop-color="#1f7fa0"/></linearGradient></defs></svg>`;
    $("#mc-pos").innerHTML = svg;
    // --- live probabilities
    const sum = (a, b) => { let s = 0; for (let i = a; i < b; i++) s += MC.counts[i]; return s / k; };
    const fin = lg.sim.summary[S.team]; // [exp_pts,p10,p90,exp_pos,title,cl,top4,top6,europe,rel,...]
    const items = [["Title", sum(0, 1), fin[4]], ["Champions L.", sum(0, cl), fin[5]], ["Top 4", sum(0, 4), fin[6]], ["Top 6", sum(0, 6), fin[7]], ["Relegation", sum(T - rel, T), fin[9]]];
    $("#mc-probs").innerHTML = items.map(([l, v, f]) => `<div class="stat"><b>${pct1(v)}</b><span>${l}</span><small>final: ${pct1(f)}</small></div>`).join("");
    // --- points histogram
    const n = MC.pc.length, W2 = 560, H2 = 270, bw2 = (W2 - ml - 8) / n, f2 = Array.from(MC.pc, (c) => c / k), y2 = Math.max(0.04, ...f2) * 1.12;
    const Y2 = (v) => mt + (H2 - mt - mb) * (1 - v / y2);
    let s2 = `<svg viewBox="0 0 ${W2} ${H2}" width="100%" role="img" aria-label="Distribution of final points">`;
    for (let g = 0; g <= 4; g++) { const v = y2 * g / 4; s2 += `<line x1="${ml}" x2="${W2 - 8}" y1="${Y2(v)}" y2="${Y2(v)}" stroke="#1f2c4a"/><text x="${ml - 6}" y="${Y2(v) + 3}" fill="#93a1c4" font-size="10" text-anchor="end">${Math.round(v * 100)}%</text>`; }
    f2.forEach((v, i) => { s2 += `<rect x="${ml + i * bw2 + 0.5}" y="${Y2(v)}" width="${Math.max(bw2 - 1, 1)}" height="${H2 - mb - Y2(v)}" fill="${C.magenta}" opacity=".85"/>`; if (i % Math.ceil(n / 9) === 0) s2 += `<text x="${ml + i * bw2 + bw2 / 2}" y="${H2 - 12}" fill="#93a1c4" font-size="10" text-anchor="middle">${MC.pmin + i}</text>`; });
    const mx = ml + (fin[0] - MC.pmin + 0.5) * bw2; s2 += `<line x1="${mx}" x2="${mx}" y1="${mt}" y2="${H2 - mb}" stroke="${C.amber}" stroke-dasharray="4 3"/><text x="${mx + 4}" y="${mt + 12}" fill="${C.amber}" font-size="11">expected ${fin[0].toFixed(1)}</text></svg>`;
    $("#mc-pts").innerHTML = s2;
    $("#mc-pts-note").textContent = `Final-points forecast over all 10,000 simulations: ${fin[0].toFixed(1)} on average, 80 % between ${Math.round(fin[1])} and ${Math.round(fin[2])} (points now: ${fin[10]} after ${fin[11]} matches).`;
    // --- ticker
    if (MC.last >= 0) {
      const row = MC.last * d.T; let champ = 0; for (let i = 0; i < T; i++) if (d.pos[row + i] === 1) champ = i;
      $("#mc-ticker").innerHTML = `simulation <b>#${(MC.last + 1).toLocaleString("en-GB")}</b> → champion <b>${d.teams[champ]}</b> · <em>${S.team}</em> finishes <b>${ord(d.pos[row + ti])}</b> with <b>${d.pts[row + ti]}</b> points`;
    } else $("#mc-ticker").textContent = "press play to start the replay";
  }
  function mcLoop() {
    if (!MC.playing) return;
    mcStep(SPEEDS[+$("#mc-speed").value]); mcDraw();
    if (MC.k >= mcData().N) { MC.playing = false; $("#mc-play").textContent = "▶ Replay"; return; }
    MC.raf = requestAnimationFrame(mcLoop);
  }
  function renderTable() {
    const lg = D.leagues[S.league], rows = Object.entries(lg.sim.summary).sort((a, b) => a[1][3] - b[1][3]);
    const bar = (v, c) => `<span class="cellbar" style="width:${Math.round(v * 60)}px;background:${c}"></span>${pct1(v)}`;
    $("#mc-table").innerHTML = `<table><thead><tr><th>Team</th><th>Pts now</th><th>Exp. pts [80 %]</th><th>Title</th><th>Champions L.</th><th>Top 4</th><th>Top 6</th><th>Relegation</th></tr></thead><tbody>` +
      rows.map(([t, s]) => `<tr class="${t === S.team ? "sel" : ""}"><td>${t}</td><td>${s[10]}/${s[11]}</td><td>${s[0].toFixed(1)} [${Math.round(s[1])}–${Math.round(s[2])}]</td><td>${bar(s[4], C.amber)}</td><td>${bar(s[5], C.cyan)}</td><td>${bar(s[6], C.cyan)}</td><td>${bar(s[7], C.violet)}</td><td>${bar(s[9], C.magenta)}</td></tr>`).join("") + "</tbody></table>";
  }
  function renderMC() { mcReset(); renderTable(); }

  /* ------------------------------------------------------------------ ARENA */
  const MODELS = [["Class frequency (naive)", "Naive frequencies", C.grey], ["Elo (LogReg)", "Elo", C.amber], ["LogReg (final)", "Logistic regression ★", C.cyan],
    ["Gradient boosting", "Gradient boosting", C.green], ["MLP deep (5 seeds)", "Neural network", C.magenta], ["Ensemble LogReg+GB+MLP", "Ensemble", C.violet], ["Bookmaker Bet365 (reference)", "Bookmaker (Bet365)", C.white]];
  const AR = { t: 0, playing: false, raf: 0, start: 10 };
  function arDraw() {
    const cv = $("#ar-canvas"), a = D.arena, dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = 430;
    cv.width = w * dpr; cv.height = h * dpr; const g = cv.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
    const ml = 52, mr = 14, mt = 14, mb = 34, N = a.n.length, ymin = 0.925, ymax = 1.10;
    const X = (i) => ml + ((i - AR.start) / (N - 1 - AR.start)) * (w - ml - mr), Y = (v) => mt + (h - mt - mb) * (1 - (Math.min(Math.max(v, ymin), ymax) - ymin) / (ymax - ymin));
    g.font = "11px ui-monospace, Menlo, monospace"; g.fillStyle = "#93a1c4"; g.strokeStyle = "#1f2c4a"; g.lineWidth = 1;
    for (let v = ymin; v <= ymax + 1e-9; v += 0.025) { g.beginPath(); g.moveTo(ml, Y(v)); g.lineTo(w - mr, Y(v)); g.stroke(); g.textAlign = "right"; g.fillText(v.toFixed(3), ml - 6, Y(v) + 4); }
    g.textAlign = "center"; let lastYear = "";
    a.dates.forEach((d, i) => { if (i < AR.start) return; const ym = d.slice(0, 7); if ((ym.endsWith("-01") || ym.endsWith("-07")) && ym !== lastYear) { lastYear = ym; g.fillText(ym, X(i), h - 12); g.beginPath(); g.moveTo(X(i), mt); g.lineTo(X(i), h - mb); g.strokeStyle = "rgba(31,44,74,.5)"; g.stroke(); } });
    g.save(); g.beginPath(); g.rect(ml, mt, w - ml - mr, h - mt - mb); g.clip();
    MODELS.forEach(([key, , col]) => {
      const ys = a.curves[key]; g.strokeStyle = col; g.lineWidth = key.startsWith("Bookmaker") ? 3 : key.startsWith("LogReg") ? 2.6 : 1.8; g.shadowColor = col; g.shadowBlur = key.startsWith("Bookmaker") || key.startsWith("LogReg") ? 10 : 4; g.beginPath();
      for (let i = AR.start; i <= AR.t; i++) i === AR.start ? g.moveTo(X(i), Y(ys[i])) : g.lineTo(X(i), Y(ys[i])); g.stroke();
      g.fillStyle = col; g.beginPath(); g.arc(X(AR.t), Y(ys[AR.t]), 3.5, 0, 7); g.fill();
    });
    g.restore();
    $("#ar-n").textContent = a.n[AR.t].toLocaleString("en-GB"); $("#ar-date").textContent = a.dates[AR.t]; $("#ar-scrub").value = AR.t;
    $("#ar-legend").innerHTML = MODELS.map(([k, name, col]) => `<span><i style="background:${col}"></i>${name} <b>${a.curves[k][AR.t].toFixed(4)}</b></span>`).join("");
  }
  function arLoop() { if (!AR.playing) return; AR.t = Math.min(AR.t + (REDUCED ? 999 : 1), D.arena.n.length - 1); arDraw(); if (AR.t >= D.arena.n.length - 1) { AR.playing = false; $("#ar-play").textContent = "▶ Replay"; return; } AR.raf = requestAnimationFrame(arLoop); }
  function renderArena() {
    const a = D.arena; $("#ar-scrub").max = a.n.length - 1; $("#ar-scrub").min = AR.start;
    if (!AR.init) { AR.t = a.n.length - 1; AR.init = true; }
    arDraw();
    $("#ar-table").innerHTML = `<table><thead><tr><th>Model</th><th>Log loss</th><th>95 % interval</th><th>Brier</th><th>Accuracy</th><th>Verdict</th></tr></thead><tbody>` + MODELS.map(([k, name]) => {
      const f = a.final[k]; const v = { "Class frequency (naive)": "reference", "Elo (LogReg)": "beaten by ML", "LogReg (final)": "CHOSEN: simplest, tied for best", "Gradient boosting": "tied, more complex", "MLP deep (5 seeds)": "tied, more complex", "Ensemble LogReg+GB+MLP": "marginally better, not retained", "Bookmaker Bet365 (reference)": "better than all of ours" }[k];
      return `<tr class="${k === "LogReg (final)" ? "sel" : ""}"><td>${name}</td><td>${f.log_loss.toFixed(4)}</td><td>${f.lo.toFixed(3)}–${f.hi.toFixed(3)}</td><td>${f.brier.toFixed(4)}</td><td>${(f.accuracy * 100).toFixed(1)}%</td><td style="font-family:var(--sans)">${v}</td></tr>`;
    }).join("") + "</tbody></table>";
  }

  /* ------------------------------------------------------------------ MODEL TAB */
  function renderModel() {
    $("#pipeline").innerHTML = [["01", "Data", "29,239 matches, 5 leagues, 2010→2026, plus 656k player valuations."], ["02", "Features", "Elo, form, shots, squad value — built only from the past (leak test)."],
      ["03", "Model", "Logistic regression, refit every season, walk-forward."], ["04", "Probabilities", "P(win), P(draw), P(loss) with uncertainty intervals."], ["05", "Simulation", "10,000 seasons → title, Europe, relegation odds."]]
      .map(([n, t, p]) => `<div class="step"><span class="no">${n}</span><h3>${t}</h3><p>${p}</p></div>`).join("");
    nnStart();
  }
  const NN = { raf: 0, t: 0 };
  function nnStart() {
    cancelAnimationFrame(NN.raf); const cv = $("#nn-canvas"); if (!cv) return;
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = 260; cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext("2d"), layers = [[14, "100 inputs"], [9, "64"], [6, "32"], [3, "3 outputs"]], xs = layers.map((_, i) => 50 + i * (w - 100) / 3);
    const nodes = layers.map(([n], li) => Array.from({ length: n }, (_, k) => [xs[li], 28 + (k + 0.5) * (h - 56) / n]));
    const rnd = (() => { let s = 7; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();
    const pulses = Array.from({ length: 26 }, () => ({ l: Math.floor(rnd() * 3), a: Math.floor(rnd() * 99), b: Math.floor(rnd() * 99), p: rnd() }));
    function frame() {
      g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
      g.lineWidth = 1; for (let l = 0; l < 3; l++) nodes[l].forEach((a) => nodes[l + 1].forEach((b) => { g.strokeStyle = "rgba(53,224,255,.07)"; g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke(); }));
      pulses.forEach((p) => { p.p += REDUCED ? 0 : 0.012; if (p.p > 1) { p.p = 0; p.l = Math.floor(rnd() * 3); p.a = Math.floor(rnd() * 99); p.b = Math.floor(rnd() * 99); }
        const A = nodes[p.l][p.a % nodes[p.l].length], B = nodes[p.l + 1][p.b % nodes[p.l + 1].length], x = A[0] + (B[0] - A[0]) * p.p, y = A[1] + (B[1] - A[1]) * p.p;
        g.fillStyle = p.l % 2 ? C.magenta : C.cyan; g.shadowColor = g.fillStyle; g.shadowBlur = 10; g.beginPath(); g.arc(x, y, 2.3, 0, 7); g.fill(); g.shadowBlur = 0; });
      nodes.forEach((layer, li) => layer.forEach((n) => { g.fillStyle = "#0d1424"; g.strokeStyle = li === 3 ? C.amber : C.cyan; g.lineWidth = 1.6; g.beginPath(); g.arc(n[0], n[1], 5, 0, 7); g.fill(); g.stroke(); }));
      g.fillStyle = "#93a1c4"; g.font = "11px ui-monospace, Menlo, monospace"; g.textAlign = "center"; layers.forEach(([, lab], li) => g.fillText(lab, xs[li], h - 6));
      ["away", "draw", "home"].forEach((t, i) => { g.textAlign = "left"; g.fillStyle = C.amber; g.fillText(t, xs[3] + 12, nodes[3][i][1] + 4); });
      if (S.tab === "model") NN.raf = requestAnimationFrame(frame);
    }
    frame();
  }

  /* ------------------------------------------------------------------ wiring */
  function showTab(t) {
    S.tab = t; $$(".tab").forEach((s) => s.classList.toggle("on", s.id === "tab-" + t)); $$(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === t));
    $(".picker").style.visibility = t === "arena" || t === "model" ? "hidden" : "visible";
    if (t === "lab") renderLab(); if (t === "mc") renderMC(); if (t === "arena") renderArena(); if (t === "model") renderModel();
    history.replaceState(null, "", "#" + t);
  }
  function onTeamChange() { resetMatchDefaults(); showTab(S.tab); }
  function init() {
    fillLeagues(); fillTeams(); resetMatchDefaults();
    $("#league").addEventListener("change", (e) => { S.league = e.target.value; S.team = teams()[0]; if (S.league === "F1") S.team = "Lyon"; fillTeams(); onTeamChange(); });
    $("#team").addEventListener("change", (e) => { S.team = e.target.value; onTeamChange(); });
    $$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
    $("#opp").addEventListener("change", (e) => { S.opp = e.target.value; renderSimulator(); });
    $("#v-home").addEventListener("click", () => { S.venue = "home"; renderSimulator(); }); $("#v-away").addEventListener("click", () => { S.venue = "away"; renderSimulator(); });
    $("#replay-wf").addEventListener("click", animateBars);
    $("#mc-play").addEventListener("click", () => { if (MC.k >= mcData().N) mcReset(false); MC.playing = !MC.playing; $("#mc-play").textContent = MC.playing ? "⏸ Pause" : "▶ Play"; if (MC.playing) mcLoop(); });
    $("#mc-skip").addEventListener("click", () => { MC.playing = false; cancelAnimationFrame(MC.raf); mcStep(1e9); mcDraw(); $("#mc-play").textContent = "▶ Replay"; });
    $("#mc-reset").addEventListener("click", () => mcReset());
    $("#ar-play").addEventListener("click", () => { if (AR.t >= D.arena.n.length - 1) AR.t = AR.start; AR.playing = !AR.playing; $("#ar-play").textContent = AR.playing ? "⏸ Pause" : "▶ Play"; if (AR.playing) arLoop(); });
    $("#ar-reset").addEventListener("click", () => { AR.playing = false; cancelAnimationFrame(AR.raf); AR.t = AR.start; $("#ar-play").textContent = "▶ Play"; arDraw(); });
    $("#ar-scrub").addEventListener("input", (e) => { AR.playing = false; AR.t = +e.target.value; $("#ar-play").textContent = "▶ Play"; arDraw(); });
    window.addEventListener("resize", () => { if (S.tab === "lab") drawSpark(info().elo_series); if (S.tab === "arena") arDraw(); if (S.tab === "model") nnStart(); });
    const hash = location.hash.replace("#", ""); showTab(["lab", "mc", "arena", "model"].includes(hash) ? hash : "lab");
  }
  init();
})();
