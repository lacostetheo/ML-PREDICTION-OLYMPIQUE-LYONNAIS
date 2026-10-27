/* OL·LAB — front end. No ML runs here: every number comes from the Python pipeline (site/data.js). */
(() => {
  "use strict";
  const D = window.OLDATA;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const S = { league: "F1", team: "Lyon", tab: "home", opp: null, venue: "home", custom: {} };
  const C = { blue: "#4f9be8", indigo: "#3a1fa8", navy: "#190a50", cyan: "#5bc8ef", green: "#4caf6a", coral: "#ef6b6b", amber: "#f2a33a", yellow: "#f5c542",
    violet: "#7c5cff", grey: "#9aa3c0", muted: "#8683ab", grid: "#e8ebf4", ink: "#241c5a", draw: "#cdd2e8" };
  const LEAGUE_ORDER = ["E0", "SP1", "I1", "D1", "F1"];

  const pct = (x) => Math.round(x * 100) + "%";
  const pct1 = (x) => (x * 100).toFixed(1) + "%";
  const range = (ci) => `${Math.round(ci[0] * 100)}–${Math.round(ci[1] * 100)}%`;
  const ord = (n) => n + (["th", "st", "nd", "rd"][(n % 100 > 10 && n % 100 < 14) ? 0 : (n % 10 < 4 ? n % 10 : 0)]);
  const fmtDate = (d, t) => new Date(d + "T12:00:00").toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" }) + (t ? " · " + t : "");
  const teams = () => D.leagues[S.league].teams;
  const info = () => D.leagues[S.league].info[S.team];
  const sumOf = (L, t) => D.leagues[L].sim.summary[t]; // [exp_pts,p10,p90,exp_pos,title,cl,top4,top6,europe,rel,pts_now,played]

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
      (labels ? `<div class="plegend"><span><i style="background:${C.blue}"></i>${labels[0]} wins</span><span><i style="background:${C.draw}"></i>Draw</span><span><i style="background:${C.coral}"></i>${labels[1]} wins</span></div>` : "");
  }
  function lightUp(root) { const on = () => $$(".pbar", root).forEach((b) => b.classList.add("on")); REDUCED ? on() : requestAnimationFrame(() => requestAnimationFrame(on)); }
  function goTo(league, team, tab) { S.league = league; S.team = team; $("#league").value = league; fillTeams(); resetMatchDefaults(); showTab(tab || "lab"); window.scrollTo(0, 0); }

  /* ------------------------------------------------------------------ pickers */
  function fillLeagues() {
    $("#league").innerHTML = Object.entries(D.leagues).map(([k, v]) => `<option value="${k}">${v.name}</option>`).join("");
    $("#league").value = S.league;
  }
  function fillTeams() { $("#team").innerHTML = teams().map((t) => `<option>${t}</option>`).join(""); $("#team").value = S.team; }
  function resetMatchDefaults() {
    const up = S.league === "F1" ? (D.calendar_f1[S.team] || [])[0] : null, nx = up || D.next[S.league][S.team];
    S.opp = nx.home === S.team ? nx.away : nx.home;
    S.venue = nx.home === S.team ? "home" : "away";
  }

  /* ------------------------------------------------------------------ DASHBOARD */
  const ICON = {
    chart: '<svg viewBox="0 0 24 24"><path d="M4 20V10h3.5v10zm6.25 0V4h3.5v16zM16.5 20v-7H20v7z"/></svg>',
    target: '<svg viewBox="0 0 24 24"><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm0 4a6 6 0 1 1 0 12 6 6 0 0 1 0-12zm0 3a3 3 0 1 0 0 6 3 3 0 0 0 0-6z"/></svg>',
    ball: '<svg viewBox="0 0 24 24"><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm0 5l3.5 2.5-1.3 4h-4.4l-1.3-4zM5 9l2 .6M19 9l-2 .6"/></svg>',
    pie: '<svg viewBox="0 0 24 24"><path d="M11 3a9 9 0 1 0 10 10h-10zM13 2v9h9A9 9 0 0 0 13 2z"/></svg>',
  };
  function wave(series, w = 220, h = 56) {
    const v = series.filter((x) => x !== null).slice(-70); if (v.length < 3) return "";
    const lo = Math.min(...v), hi = Math.max(...v), X = (i) => (i / (v.length - 1)) * w, Y = (x) => h - 6 - ((x - lo) / (hi - lo || 1)) * (h - 18);
    let d = `M0,${h} `; v.forEach((x, i) => (d += `L${X(i).toFixed(1)},${Y(x).toFixed(1)} `)); d += `L${w},${h} Z`;
    let ln = ""; v.forEach((x, i) => (ln += `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(x).toFixed(1)} `));
    return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><path d="${d}" fill="rgba(255,255,255,.16)"/><path d="${ln}" fill="none" stroke="#fff" stroke-width="2.2" stroke-linejoin="round"/></svg>`;
  }
  function top3(L) {
    const rows = Object.entries(D.leagues[L].sim.summary).sort((a, b) => a[1][3] - b[1][3]).slice(0, 3);
    return rows.map(([t, s]) => ({ t, s, p3: D.leagues[L].sim.pos_probs[t].slice(0, 3).reduce((a, b) => a + b, 0) }));
  }
  function renderHome() {
    $("#asof").textContent = `Snapshot ${D.as_of} · 10,000 simulated seasons per league · walk-forward logistic regression`;
    const fin = D.arena.final["LogReg (final)"], bk = D.arena.final["Bookmaker Bet365 (reference)"];
    const ol = sumOf("F1", "Lyon"), olp = D.leagues.F1.sim.pos_probs.Lyon, gap = fin.log_loss - bk.log_loss;
    const mx = Math.max(...olp);
    $("#kpis").innerHTML = `
      <div class="card"><div class="kpi"><div><div class="val">29,239</div><div class="lab">Matches analysed</div></div><div class="tile t-yellow">${ICON.chart}</div></div><div class="foot flat">5 leagues · 2010 → 2026</div></div>
      <div class="card"><div class="kpi"><div><div class="val">${fin.log_loss.toFixed(3)}</div><div class="lab">Test log loss (lower = better)</div></div><div class="tile t-cyan">${ICON.target}</div></div><div class="foot down">▲ ${gap.toFixed(3)} behind the bookmaker (${bk.log_loss.toFixed(3)})</div></div>
      <div class="card"><div class="kpi"><div><div class="val">${ol[0].toFixed(1)}</div><div class="lab">OL expected points · Ligue 1</div></div><div class="tile t-green">${ICON.ball}</div></div><div class="foot up">80 % interval ${Math.round(ol[1])}–${Math.round(ol[2])}</div></div>
      <div class="card"><div class="kpi"><div><div class="val">${pct(ol[6])}</div><div class="lab">OL chance of top 4</div></div><div class="tile t-violet">${ICON.pie}</div></div>
        <div class="minibars" aria-hidden="true">${olp.map((v) => `<i style="height:${Math.max(6, v / mx * 100)}%"></i>`).join("")}</div></div>`;

    $("#top3").innerHTML = LEAGUE_ORDER.map((L, i) => {
      const r = top3(L), champ = r[0];
      return `<article class="card lg c${i}"><div class="head"><div class="lname">${D.leagues[L].name}</div><div class="champ">${champ.t}</div>
        <div class="cprob">projected champion · title ${pct(champ.s[4])}</div>${wave(D.leagues[L].info[champ.t].elo_series)}</div>
        <div class="body">${r.map((x, k) => `<div class="row" data-go="${L}|${x.t}"><span class="medal m${k + 1}">${k + 1}</span><div class="tn">${x.t}<span class="ts">${x.s[0].toFixed(0)} pts · title ${pct(x.s[4])}</span></div><span class="pill" title="chance of a top-3 finish">${pct(x.p3)}</span></div>`).join("")}</div></article>`;
    }).join("");

    // title race of the selected league
    const lg = D.leagues[S.league], rows = Object.entries(lg.sim.summary).sort((a, b) => a[1][3] - b[1][3]).slice(0, 8);
    $("#race-title").innerHTML = `Title race — ${lg.name} <span class="chip">top 8 by expected position</span>`;
    const W = 720, H = 250, ml = 38, mb = 52, mt = 10, bw = (W - ml - 8) / rows.length;
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Title and top-4 probabilities">`;
    for (let g = 0; g <= 4; g++) { const y = mt + (H - mt - mb) * (1 - g / 4); svg += `<line x1="${ml}" x2="${W - 8}" y1="${y}" y2="${y}" stroke="${C.grid}"/><text x="${ml - 6}" y="${y + 3}" fill="${C.muted}" font-size="10" text-anchor="end">${g * 25}</text>`; }
    rows.forEach(([t, s], i) => {
      const x0 = ml + i * bw + bw * 0.18, w = bw * 0.30, Y = (v) => mt + (H - mt - mb) * (1 - v);
      svg += `<rect x="${x0}" y="${Y(s[4])}" width="${w}" height="${H - mb - Y(s[4])}" rx="3" fill="${C.blue}" opacity=".75"><title>${t}: title ${pct1(s[4])}</title></rect>`;
      svg += `<rect x="${x0 + w + 3}" y="${Y(s[6])}" width="${w}" height="${H - mb - Y(s[6])}" rx="3" fill="${C.indigo}"><title>${t}: top 4 ${pct1(s[6])}</title></rect>`;
      svg += `<text x="${x0 + w}" y="${H - mb + 16}" fill="${C.ink}" font-size="10.5" text-anchor="middle" font-weight="600">${t.length > 11 ? t.slice(0, 10) + "…" : t}</text>`;
    });
    svg += `<g font-size="10.5" fill="${C.muted}"><rect x="${ml}" y="${H - 16}" width="10" height="10" rx="2" fill="${C.blue}" opacity=".75"/><text x="${ml + 15}" y="${H - 7}">win the league</text><rect x="${ml + 120}" y="${H - 16}" width="10" height="10" rx="2" fill="${C.indigo}"/><text x="${ml + 135}" y="${H - 7}">finish top 4</text><text x="${W - 8}" y="${H - 7}" text-anchor="end">probability, %</text></g></svg>`;
    $("#race").innerHTML = svg;

    // upcoming round of the selected league
    const seen = new Set(), fx = [];
    Object.values(D.next[S.league]).forEach((m) => { const k = m.home + "|" + m.away; if (!seen.has(k)) { seen.add(k); fx.push(m); } });
    fx.sort((a, b) => (a.date + a.time).localeCompare(b.date + b.time));
    $("#up-day").textContent = `${lg.name} · ${fmtDate(fx[0].date)} → ${fmtDate(fx[fx.length - 1].date)}`;
    $("#upcoming").innerHTML = fx.map((m) => {
      const [a, d, h] = m.p, best = h >= a && h >= d ? "u" : a >= d ? "d" : "n";
      return `<div class="up-item" data-go="${S.league}|${m.home}"><div class="up-ico ${best}">${best === "u" ? "↑" : best === "d" ? "↓" : "="}</div><div><div class="ut">${m.home} vs ${m.away}</div><div class="us">${pct(h)} / ${pct(d)} / ${pct(a)} · ${fmtDate(m.date)}</div></div></div>`;
    }).join("");

    // Europe: qualification odds from the domestic simulations
    const all = [];
    LEAGUE_ORDER.forEach((L) => Object.entries(D.leagues[L].sim.summary).forEach(([t, s]) => all.push({ L, t, cl: s[5], eu: Math.max(0, s[8] - s[5]) })));
    const bar = (v, c) => `<span class="cellbar" style="width:${Math.round(v * 70)}px;background:${c}"></span>${pct(v)}`;
    const tbl = (rows, key, color) => `<table><thead><tr><th>Team</th><th>League</th><th>Probability</th></tr></thead><tbody>${rows.map((r) => `<tr data-go="${r.L}|${r.t}" style="cursor:pointer"><td>${r.t}</td><td>${D.leagues[r.L].name}</td><td class="num">${bar(r[key], color)}</td></tr>`).join("")}</tbody></table>`;
    const U = OLDATA.ucl;
    if (U) {
      const pc = (v) => (v * 100).toFixed(v < 0.1 ? 1 : 0) + " %";
      $("#ucl-win").innerHTML = `<table><thead><tr><th>Club</th><th>Top 8</th><th>Semi</th><th>Final</th><th>Winner</th></tr></thead><tbody>` +
        U.rows.map((r) => `<tr><td>${r.team}</td><td>${pc(r.top8)}</td><td>${pc(r.SF)}</td><td>${pc(r.Final)}</td><td><b>${pc(r.Winner)}</b></td></tr>`).join("") + `</tbody></table>`;
      $("#ucl-note").textContent = `${U.played} of ${U.total} league-phase matches in the data. No clear favourite: the six leading clubs are within a few points of each other.`;
    }
    $("#ucl").innerHTML = tbl([...all].sort((a, b) => b.cl - a.cl).slice(0, 14), "cl", C.blue);
    $("#uel").innerHTML = tbl([...all].sort((a, b) => b.eu - a.eu).slice(0, 14), "eu", C.violet);
  }

  /* ------------------------------------------------------------------ MATCH LAB */
  function renderTeamCard() {
    const i = info(), n = teams().length, sv = i.sv === null ? "n/a" : "×" + Math.exp(i.sv).toFixed(2);
    $("#teamcard").innerHTML = `<div class="name">${S.team}</div><div class="sub">${D.leagues[S.league].name} · snapshot ${D.as_of}</div>
      <div class="stats">
        <div class="stat"><b>${Math.round(i.elo)}</b><span>Elo · ${ord(i.elo_rank)}/${n}</span></div>
        <div class="stat"><b>${sv}</b><span>Squad value vs league avg</span></div>
        <div class="stat"><b>${i.pts10.toFixed(2)}</b><span>Points / match (last 10)</span></div>
        <div class="stat"><b>${i.gf10.toFixed(1)}–${i.ga10.toFixed(1)}</b><span>Goals for–against / match</span></div>
      </div><canvas id="spark" height="90" aria-label="Elo rating history"></canvas><div class="note">Elo rating since 2010 (monthly)</div>`;
    drawSpark(i.elo_series);
  }
  function drawSpark(series) {
    const cv = $("#spark"); if (!cv) return;
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = 90;
    cv.width = w * dpr; cv.height = h * dpr; const g = cv.getContext("2d"); g.scale(dpr, dpr);
    const vals = series.filter((v) => v !== null); if (!vals.length) return;
    const lo = Math.min(...vals) - 20, hi = Math.max(...vals) + 20;
    const X = (k) => 6 + (k / (series.length - 1)) * (w - 12), Y = (v) => h - 8 - ((v - lo) / (hi - lo)) * (h - 20);
    g.strokeStyle = C.grid; g.lineWidth = 1; [lo + 20, (lo + hi) / 2, hi - 20].forEach((v) => { g.beginPath(); g.moveTo(0, Y(v)); g.lineTo(w, Y(v)); g.stroke(); });
    const grad = g.createLinearGradient(0, 0, w, 0); grad.addColorStop(0, C.blue); grad.addColorStop(1, C.violet);
    g.strokeStyle = grad; g.lineWidth = 2; g.lineJoin = "round"; g.beginPath(); let pen = false;
    series.forEach((v, k) => { if (v === null) { pen = false; return; } pen ? g.lineTo(X(k), Y(v)) : g.moveTo(X(k), Y(v)); pen = true; });
    g.stroke();
    const last = series.length - 1 - [...series].reverse().findIndex((v) => v !== null);
    g.fillStyle = C.coral; g.beginPath(); g.arc(X(last), Y(series[last]), 4, 0, 7); g.fill();
  }

  function matchCard(n, home, away, date, time, arr, market, extra, stale) {
    const t = S.team, h = home === t, opp = h ? away : home, p = persp(home, t, arr);
    const mk = market ? persp(home, t, [market[0], market[1], market[2], 0, 0, 0, 0, 0, 0]) : null;
    return `<article class="mcard ${n === 1 ? "first" : ""}"><div class="meta"><span class="badge-n ${stale ? "stale" : ""}">${stale ? "ROUND PLAYED" : n === 1 ? "NEXT" : "THEN"} · #${n}</span><span>${fmtDate(date, time)}${extra || ""}</span></div>
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
        html += real ? matchCard(k + 1, nx.home, nx.away, nx.date, nx.time, nx.p.concat(nx.ci), nx.market, ' · <b class="up">confirmed</b>')
                     : matchCard(k + 1, m.home, m.away, m.date, "", D.pairs.F1[`${m.home}|${m.away}`], null, ` · matchday ${m.matchday}`);
      });
      $("#next3-note").textContent = "From today (" + D.as_of + ") on, per the published Ligue 1 calendar (dates can change). Only a match listed by the odds source has bookmaker prices and real rest days; later matches use a neutral schedule. The further ahead, the less today's snapshot reflects reality — Elo and form move after every match.";
    } else {
      const stale = nx.date < D.as_of;
      html = matchCard(1, nx.home, nx.away, nx.date, nx.time, nx.p.concat(nx.ci), nx.market, stale ? " · played, result not in the data yet" : ' · <b class="up">confirmed</b>', stale) + customCard(2) + customCard(3);
      $("#next3-note").textContent = "Match 1 is the fixture listed by the odds source" + (stale ? " (already played before the snapshot: this is the forecast the model would have made). " : ". ") + "For this league the calendar beyond that round is not part of the data source, so matches 2–3 are yours to choose.";
    }
    $("#next3").innerHTML = html; lightUp($("#next3"));
    $$(".c-opp").forEach((s) => s.addEventListener("change", (e) => { const key = S.league + "|" + S.team; S.custom[key][+e.target.dataset.c].opp = e.target.value || undefined; renderNext3(); }));
    $$(".c-v").forEach((b) => b.addEventListener("click", () => { const key = S.league + "|" + S.team; S.custom[key][+b.dataset.c].venue = b.dataset.v; renderNext3(); }));
  }
  function renderSimulator() {
    const opts = teams().filter((x) => x !== S.team);
    $("#opp").innerHTML = opts.map((o) => `<option>${o}</option>`).join(""); $("#opp").value = S.opp;
    $("#v-home").classList.toggle("on", S.venue === "home"); $("#v-away").classList.toggle("on", S.venue === "away");
    const home = S.venue === "home" ? S.team : S.opp, away = S.venue === "home" ? S.opp : S.team;
    const arr = D.pairs[S.league][`${home}|${away}`], p = persp(home, S.team, arr), eloDiff = (S.venue === "home" ? 1 : -1) * arr[17];
    $("#match-title").innerHTML = `${home} <small>vs</small> ${away}<br><small>${S.team} playing ${S.venue} · neutral schedule (7 days' rest each)</small>`;
    $("#match-bar").innerHTML = pbar(p, true, [S.team, S.opp]); lightUp($("#match-bar"));
    const nx = D.next[S.league][S.team], isNext = nx.home === home && nx.away === away;
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
    const row = (lab, v, cls) => `<div class="wf-row ${cls || ""}"><div class="lab">${lab}</div><div class="wf-track"><div class="wf-bar ${v >= 0 ? "pos" : "neg"}" data-w="${(Math.abs(v) / max * 49).toFixed(1)}"></div></div><div class="wf-val" style="color:${v >= 0 ? C.blue : C.coral}">${v >= 0 ? "+" : ""}${v.toFixed(2)}</div></div>`;
    $("#waterfall").innerHTML = `<div class="wf-axis"><span>← favours ${S.opp}</span><span>favours ${S.team} →</span></div>` + items.map((x) => row(x.f, x.v)).join("") + row("<b>Net push</b>", total, "wf-total");
    animateBars();
  }
  function animateBars() { $$(".wf-bar").forEach((b, i) => { b.style.width = "0"; setTimeout(() => (b.style.width = b.dataset.w + "%"), REDUCED ? 0 : 120 + i * 130); }); }
  function renderLab() { renderTeamCard(); renderNext3(); renderSimulator(); $("#asof-lab").textContent = D.as_of; $("#asof2").textContent = D.as_of; }

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
    $("#mc-total").textContent = d.N.toLocaleString("en-GB"); $("#mc-team-lead").textContent = S.team + "'s";
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
    const W = 560, H = 270, ml = 38, mb = 30, mt = 10, bw = (W - ml - 8) / T;
    const freq = Array.from(MC.counts, (c) => c / k), ymax = Math.max(0.05, ...target, ...(MC.k > 30 ? freq : [0])) * 1.12;
    const Y = (v) => mt + (H - mt - mb) * (1 - v / ymax), cl = lg.slots.champions_league, eu = lg.slots.europe, rel = lg.slots.relegation;
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Distribution of final position">`;
    const band = (a, b, col, lab) => `<rect x="${ml + (a - 1) * bw}" y="${mt}" width="${(b - a + 1) * bw}" height="${H - mt - mb}" fill="${col}"/><text x="${ml + ((a - 1) + (b - a + 1) / 2) * bw}" y="${mt + 11}" fill="${C.muted}" font-size="9" text-anchor="middle">${lab}</text>`;
    svg += band(1, cl, "rgba(79,155,232,.10)", "Champions L.") + band(cl + 1, eu, "rgba(124,92,255,.08)", "Europe") + band(T - rel + 1, T, "rgba(239,107,107,.12)", "Relegation");
    for (let g = 0; g <= 4; g++) { const v = ymax * g / 4; svg += `<line x1="${ml}" x2="${W - 8}" y1="${Y(v)}" y2="${Y(v)}" stroke="${C.grid}"/><text x="${ml - 6}" y="${Y(v) + 3}" fill="${C.muted}" font-size="10" text-anchor="end">${Math.round(v * 100)}%</text>`; }
    for (let i = 0; i < T; i++) {
      const x = ml + i * bw + 2, w = bw - 4;
      svg += `<rect x="${x}" y="${Y(freq[i])}" width="${w}" height="${H - mb - Y(freq[i])}" rx="3" fill="${C.blue}"/>`;
      svg += `<rect x="${x}" y="${Y(target[i])}" width="${w}" height="${H - mb - Y(target[i])}" rx="3" fill="none" stroke="${C.amber}" stroke-width="1.6" stroke-dasharray="3 3"/>`;
      svg += `<text x="${x + w / 2}" y="${H - 12}" fill="${C.muted}" font-size="10" text-anchor="middle">${i + 1}</text>`;
    }
    $("#mc-pos").innerHTML = svg + "</svg>";
    const sum = (a, b) => { let s = 0; for (let i = a; i < b; i++) s += MC.counts[i]; return s / k; };
    const fin = lg.sim.summary[S.team];
    const items = [["Title", sum(0, 1), fin[4]], ["Champions L.", sum(0, cl), fin[5]], ["Top 4", sum(0, 4), fin[6]], ["Top 6", sum(0, 6), fin[7]], ["Relegation", sum(T - rel, T), fin[9]]];
    $("#mc-probs").innerHTML = items.map(([l, v, f]) => `<div class="stat"><b>${pct1(v)}</b><span>${l}</span><small>final: ${pct1(f)}</small></div>`).join("");
    const n = MC.pc.length, W2 = 560, H2 = 270, bw2 = (W2 - ml - 8) / n, f2 = Array.from(MC.pc, (c) => c / k), y2 = Math.max(0.04, ...f2) * 1.12;
    const Y2 = (v) => mt + (H2 - mt - mb) * (1 - v / y2);
    let s2 = `<svg viewBox="0 0 ${W2} ${H2}" width="100%" role="img" aria-label="Distribution of final points">`;
    for (let g = 0; g <= 4; g++) { const v = y2 * g / 4; s2 += `<line x1="${ml}" x2="${W2 - 8}" y1="${Y2(v)}" y2="${Y2(v)}" stroke="${C.grid}"/><text x="${ml - 6}" y="${Y2(v) + 3}" fill="${C.muted}" font-size="10" text-anchor="end">${Math.round(v * 100)}%</text>`; }
    f2.forEach((v, i) => { s2 += `<rect x="${ml + i * bw2 + 0.5}" y="${Y2(v)}" width="${Math.max(bw2 - 1, 1)}" height="${H2 - mb - Y2(v)}" fill="${C.indigo}" opacity=".85"/>`; if (i % Math.ceil(n / 9) === 0) s2 += `<text x="${ml + i * bw2 + bw2 / 2}" y="${H2 - 12}" fill="${C.muted}" font-size="10" text-anchor="middle">${MC.pmin + i}</text>`; });
    const mx = ml + (fin[0] - MC.pmin + 0.5) * bw2; s2 += `<line x1="${mx}" x2="${mx}" y1="${mt}" y2="${H2 - mb}" stroke="${C.amber}" stroke-width="1.6" stroke-dasharray="4 3"/><text x="${mx + 4}" y="${mt + 12}" fill="#b9770e" font-size="11">expected ${fin[0].toFixed(1)}</text></svg>`;
    $("#mc-pts").innerHTML = s2;
    $("#mc-pts-note").textContent = `Final-points forecast over all 10,000 simulations: ${fin[0].toFixed(1)} on average, 80 % between ${Math.round(fin[1])} and ${Math.round(fin[2])} (points now: ${fin[10]} after ${fin[11]} matches).`;
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
      rows.map(([t, s]) => `<tr class="${t === S.team ? "sel" : ""}"><td>${t}</td><td class="num">${s[10]}/${s[11]}</td><td class="num">${s[0].toFixed(1)} [${Math.round(s[1])}–${Math.round(s[2])}]</td><td class="num">${bar(s[4], C.amber)}</td><td class="num">${bar(s[5], C.blue)}</td><td class="num">${bar(s[6], C.blue)}</td><td class="num">${bar(s[7], C.violet)}</td><td class="num">${bar(s[9], C.coral)}</td></tr>`).join("") + "</tbody></table>";
  }
  function renderMC() { mcReset(); renderTable(); }

  /* ------------------------------------------------------------------ ARENA */
  const MODELS = [["Class frequency (naive)", "Naive frequencies", C.grey], ["Elo (LogReg)", "Elo", C.amber], ["LogReg (final)", "Logistic regression ★", C.blue],
    ["Gradient boosting", "Gradient boosting", C.green], ["MLP deep (5 seeds)", "Neural network", "#e0489c"], ["Ensemble LogReg+GB+MLP", "Ensemble", C.violet], ["Bookmaker Bet365 (reference)", "Bookmaker (Bet365)", C.navy]];
  const AR = { t: 0, playing: false, raf: 0, start: 10 };
  function arDraw() {
    const cv = $("#ar-canvas"), a = D.arena, dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = 430;
    cv.width = w * dpr; cv.height = h * dpr; const g = cv.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
    const ml = 52, mr = 14, mt = 14, mb = 34, N = a.n.length, ymin = 0.925, ymax = 1.10;
    const X = (i) => ml + ((i - AR.start) / (N - 1 - AR.start)) * (w - ml - mr), Y = (v) => mt + (h - mt - mb) * (1 - (Math.min(Math.max(v, ymin), ymax) - ymin) / (ymax - ymin));
    g.font = "11px ui-monospace, Menlo, monospace"; g.fillStyle = C.muted; g.strokeStyle = C.grid; g.lineWidth = 1;
    for (let v = ymin; v <= ymax + 1e-9; v += 0.025) { g.beginPath(); g.moveTo(ml, Y(v)); g.lineTo(w - mr, Y(v)); g.stroke(); g.textAlign = "right"; g.fillText(v.toFixed(3), ml - 6, Y(v) + 4); }
    g.textAlign = "center"; let lastYear = "";
    a.dates.forEach((d, i) => { if (i < AR.start) return; const ym = d.slice(0, 7); if ((ym.endsWith("-01") || ym.endsWith("-07")) && ym !== lastYear) { lastYear = ym; g.fillText(ym, X(i), h - 12); g.beginPath(); g.moveTo(X(i), mt); g.lineTo(X(i), h - mb); g.strokeStyle = "rgba(190,196,225,.5)"; g.stroke(); } });
    g.save(); g.beginPath(); g.rect(ml, mt, w - ml - mr, h - mt - mb); g.clip();
    MODELS.forEach(([key, , col]) => {
      const ys = a.curves[key]; g.strokeStyle = col; g.lineWidth = key.startsWith("Bookmaker") ? 3.2 : key.startsWith("LogReg") ? 2.8 : 1.9; g.lineJoin = "round"; g.beginPath();
      for (let i = AR.start; i <= AR.t; i++) i === AR.start ? g.moveTo(X(i), Y(ys[i])) : g.lineTo(X(i), Y(ys[i])); g.stroke();
      g.fillStyle = col; g.beginPath(); g.arc(X(AR.t), Y(ys[AR.t]), 3.6, 0, 7); g.fill();
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
    const verdict = { "Class frequency (naive)": ["reference", "grey"], "Elo (LogReg)": ["beaten by ML", "amber"], "LogReg (final)": ["CHOSEN: simplest, tied for best", "blue"], "Gradient boosting": ["tied, more complex", "grey"],
      "MLP deep (5 seeds)": ["tied, more complex", "grey"], "Ensemble LogReg+GB+MLP": ["marginally better, not retained", "grey"], "Bookmaker Bet365 (reference)": ["better than all of ours", "green"] };
    $("#ar-table").innerHTML = `<table><thead><tr><th>Model</th><th>Log loss</th><th>95 % interval</th><th>Brier</th><th>Accuracy</th><th>Verdict</th></tr></thead><tbody>` + MODELS.map(([k, name]) => {
      const f = a.final[k], v = verdict[k];
      return `<tr class="${k === "LogReg (final)" ? "sel" : ""}"><td>${name}</td><td class="num">${f.log_loss.toFixed(4)}</td><td class="num">${f.lo.toFixed(3)}–${f.hi.toFixed(3)}</td><td class="num">${f.brier.toFixed(4)}</td><td class="num">${(f.accuracy * 100).toFixed(1)}%</td><td><span class="status s-${v[1]}">${v[0]}</span></td></tr>`;
    }).join("") + "</tbody></table>";
  }

  /* ------------------------------------------------------------------ MODEL TAB */
  function renderModel() {
    $("#pipeline").innerHTML = [["01", "Data", "29,239 matches, 5 leagues, 2010→2026, plus 656k player valuations."], ["02", "Features", "Elo, form, shots, squad value — built only from the past (leak test)."],
      ["03", "Model", "Logistic regression, refit every season, walk-forward."], ["04", "Probabilities", "P(win), P(draw), P(loss) with uncertainty intervals."], ["05", "Simulation", "10,000 seasons → title, Europe, relegation odds."]]
      .map(([n, t, p]) => `<div class="step"><span class="no">${n}</span><h3>${t}</h3><p>${p}</p></div>`).join("");
    nnStart();
  }
  const NN = { raf: 0 };
  function nnStart() {
    cancelAnimationFrame(NN.raf); const cv = $("#nn-canvas"); if (!cv) return;
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = 260; cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext("2d"), layers = [[14, "100 inputs"], [9, "64"], [6, "32"], [3, "3 outputs"]], xs = layers.map((_, i) => 50 + i * (w - 100) / 3);
    const nodes = layers.map(([n], li) => Array.from({ length: n }, (_, k) => [xs[li], 28 + (k + 0.5) * (h - 56) / n]));
    const rnd = (() => { let s = 7; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();
    const pulses = Array.from({ length: 26 }, () => ({ l: Math.floor(rnd() * 3), a: Math.floor(rnd() * 99), b: Math.floor(rnd() * 99), p: rnd() }));
    function frame() {
      g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h); g.lineWidth = 1;
      for (let l = 0; l < 3; l++) nodes[l].forEach((a) => nodes[l + 1].forEach((b) => { g.strokeStyle = "rgba(58,31,168,.07)"; g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke(); }));
      pulses.forEach((p) => { p.p += REDUCED ? 0 : 0.012; if (p.p > 1) { p.p = 0; p.l = Math.floor(rnd() * 3); p.a = Math.floor(rnd() * 99); p.b = Math.floor(rnd() * 99); }
        const A = nodes[p.l][p.a % nodes[p.l].length], B = nodes[p.l + 1][p.b % nodes[p.l + 1].length];
        g.fillStyle = p.l % 2 ? C.coral : C.blue; g.beginPath(); g.arc(A[0] + (B[0] - A[0]) * p.p, A[1] + (B[1] - A[1]) * p.p, 2.4, 0, 7); g.fill(); });
      nodes.forEach((layer, li) => layer.forEach((n) => { g.fillStyle = "#fff"; g.strokeStyle = li === 3 ? C.amber : C.indigo; g.lineWidth = 1.7; g.beginPath(); g.arc(n[0], n[1], 5, 0, 7); g.fill(); g.stroke(); }));
      g.fillStyle = C.muted; g.font = "11px ui-monospace, Menlo, monospace"; g.textAlign = "center"; layers.forEach(([, lab], li) => g.fillText(lab, xs[li], h - 6));
      ["away", "draw", "home"].forEach((t, i) => { g.textAlign = "left"; g.fillStyle = C.ink; g.fillText(t, xs[3] + 12, nodes[3][i][1] + 4); });
      if (S.tab === "model") NN.raf = requestAnimationFrame(frame);
    }
    frame();
  }

  /* ------------------------------------------------------------------ wiring */
  function showTab(t) {
    S.tab = t; $$(".tab").forEach((s) => s.classList.toggle("on", s.id === "tab-" + t)); $$(".nav button").forEach((b) => b.classList.toggle("on", b.dataset.tab === t));
    $("#picker").style.visibility = t === "arena" || t === "model" ? "hidden" : "visible";
    if (t === "home") renderHome(); if (t === "lab") renderLab(); if (t === "mc") renderMC(); if (t === "arena") renderArena(); if (t === "model") renderModel();
    history.replaceState(null, "", "#" + t);
  }
  function onTeamChange() { resetMatchDefaults(); showTab(S.tab); }
  function init() {
    fillLeagues(); fillTeams(); resetMatchDefaults();
    $("#league").addEventListener("change", (e) => { S.league = e.target.value; S.team = S.league === "F1" ? "Lyon" : teams()[0]; fillTeams(); onTeamChange(); });
    $("#team").addEventListener("change", (e) => { S.team = e.target.value; onTeamChange(); });
    $$(".nav button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
    document.addEventListener("click", (e) => { const el = e.target.closest("[data-go]"); if (el) { const [L, t] = el.dataset.go.split("|"); goTo(L, t, "lab"); } });
    $("#hide-menu").addEventListener("click", () => { const c = $("#shell").classList.toggle("collapsed"); $("#hide-menu span").textContent = c ? "Show Menu" : "Hide Menu"; setTimeout(() => window.dispatchEvent(new Event("resize")), 300); });
    if (window.innerWidth < 1000) { $("#shell").classList.add("collapsed"); $("#hide-menu span").textContent = "Show Menu"; }
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
    const hash = location.hash.replace("#", ""); showTab(["home", "lab", "mc", "arena", "model"].includes(hash) ? hash : "home");
  }
  init();
})();
