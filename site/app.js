/* OL·LAB — front end. Nothing is computed by a model here: every number comes from the Python pipeline (data.js). */
(() => {
  "use strict";
  const D = window.OLDATA;
  const FOCUS = "Lyon";                       // the case study: always drawn in the accent colour
  const LEAGUES = ["F1", "E0", "SP1", "I1", "D1"];
  const COMPS = [...LEAGUES, "ucl", "uel"];
  const INK = "#16181D", INK3 = "#626672", GRAY = "#CFCBC0", MID = "#8E919B", ACCENT = "#C8321E", RULE = "#DDD9CE", PAPER = "#FBF9F4";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const nm = (t) => esc(D.display[t] || t);
  const isLeague = (c) => Object.prototype.hasOwnProperty.call(D.leagues, c);
  const compName = (c) => (isLeague(c) ? D.leagues[c].name : D.europe[c].name);
  const pc = (x, d = 0) => (100 * x).toFixed(d) + "%";
  const pr = (x) => Math.round(100 * x);
  const ORD = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
    "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth"];
  const ordWord = (n) => ORD[n] || n + "th";
  const ordNum = (n) => n + (["th", "st", "nd", "rd"][(n % 100 > 10 && n % 100 < 14) ? 0 : (n % 10 < 4 ? n % 10 : 0)]);
  const dt = (s) => new Date(s + "T12:00:00");
  const fm = (s, o) => dt(s).toLocaleDateString("en-GB", o);
  const dDay = (s) => fm(s, { day: "numeric", month: "short" });
  const dLong = (s) => fm(s, { weekday: "long", day: "numeric", month: "long", year: "numeric" });
  const dHead = (s) => fm(s, { weekday: "long", day: "numeric", month: "long" });
  const daysBetween = (a, b) => Math.round((dt(b) - dt(a)) / 864e5);
  const leagueOf = (team) => LEAGUES.find((L) => D.leagues[L].teams.includes(team));
  const fixturesOf = (c) => (isLeague(c) ? D.fixtures[c] : D.europe[c].fixtures);
  const cmpFx = (x, y) => (x.d + (x.t || "99")).localeCompare(y.d + (y.t || "99"));
  const teamFixtures = (team) => COMPS.flatMap((c) => fixturesOf(c).filter((m) => m.h === team || m.a === team).map((m) => ({ ...m, comp: c }))).sort(cmpFx);
  const showTime = (m) => m.t && daysBetween(D.as_of, m.d) <= 10;       // later kick-off times are placeholders
  const euroOf = (team) => ["ucl", "uel"].map((c) => ({ comp: c, row: D.europe[c].teams.find((x) => x.k === team) })).find((x) => x.row);
  const pt = (x) => (x < 0.005 ? "–" : pc(x));                      // tables: a dash instead of a meaningless 0%
  const pw = (x) => (x < 0.0005 ? "&lt;0.1%" : pc(x, 1));
  const isRight = (r) => r[7].indexOf(Math.max(...r[7])) === r[6];
  const S = { page: null, comp: "F1", round: {}, mteam: "", rlg: "" };
  const charts = [];

  /* ------------------------------------------------------------------ small components */
  function split3(p) {                                         // p = [away, draw, home]
    const [a, d, h] = p, max = Math.max(a, d, h);
    return { h, d, a, fav: max === h ? "h" : max === a ? "a" : "d" };
  }
  function p3(m, focus) {
    const { h, d, a, fav } = split3(m.p);
    const sh = m.h === focus ? "sx" : fav === "h" ? "sf" : "so", sa = m.a === focus ? "sx" : fav === "a" ? "sf" : "so", sdr = fav === "d" ? "sf" : "sd";
    const w = (x) => `style="flex:${Math.max(x, 0.001).toFixed(4)} 1 0"`;
    return `<span class="p3"><span class="nums"><span ${w(h)}${fav === "h" ? ' class="fav"' : ""}><b>${pr(h)}%</b></span><span ${w(d)}${fav === "d" ? ' class="fav"' : ""}><b>${pr(d)}%</b></span><span ${w(a)}${fav === "a" ? ' class="fav"' : ""}><b>${pr(a)}%</b></span></span>` +
      `<span class="seg"><i class="${sh}" style="width:${(h * 100).toFixed(1)}%"></i><i class="${sdr}" style="width:${(d * 100).toFixed(1)}%"></i><i class="${sa}" style="width:${(a * 100).toFixed(1)}%"></i></span></span>`;
  }
  function fxRow(m, o = {}) {
    const focus = o.focus || FOCUS, comp = m.comp || o.comp, { fav } = split3(m.p);
    const time = m.late ? `<span class="late">Postponed</span>` : (showTime(m) ? m.t : "");
    const when = o.dated ? `<b>${fm(m.d, { weekday: "short", day: "numeric", month: "short" })}</b>${time ? " · " + time : ""} · ${esc(compName(comp))}` : time;
    const inner = `<span class="when">${when}</span><span class="teams"><span class="t ${fav === "h" ? "fav" : ""}">${nm(m.h)}</span><span class="t ${fav === "a" ? "fav" : ""}">${nm(m.a)}</span></span>${p3(m, focus)}`;
    if (o.expand !== false && isLeague(comp)) {
      return `<li><button class="fx" aria-expanded="false" data-k="${esc(comp + "|" + m.h + "|" + m.a)}">${inner}</button><div class="detail" hidden></div></li>`;
    }
    return `<li><div class="fx" style="cursor:default">${inner}</div></li>`;
  }
  function detail(key) {
    const [c, h, a] = key.split("|"), v = D.pairs[c][`${h}|${a}`], lg = D.leagues[c], ih = lg.info[h], ia = lg.info[a];
    const contrib = v.slice(9, 17), elo = v[17], net = contrib.reduce((s, x) => s + x, 0);
    const mx = Math.max(0.5, ...contrib.map(Math.abs), Math.abs(net));
    const bar = (x) => (x >= 0 ? `<i style="left:50%;width:${(x / mx * 50).toFixed(1)}%"></i>` : `<i class="neg" style="right:50%;width:${(-x / mx * 50).toFixed(1)}%"></i>`);
    const sg = (x) => (Math.abs(x) < 0.005 ? "0.00" : (x > 0 ? "+" : "") + x.toFixed(2));
    const order = D.families.map((f, i) => ({ f, x: contrib[i] })).sort((p, q) => Math.abs(q.x) - Math.abs(p.x));
    const why = order.map((o) => `<span>${esc(o.f)}</span><span class="tr">${bar(o.x)}</span><span class="v">${sg(o.x)}</span>`).join("") +
      `<span class="tot">Net push</span><span class="tr">${bar(net)}</span><span class="v tot">${sg(net)}</span>`;
    const sv = (i) => (i.sv === null ? "n/a" : "×" + Math.exp(i.sv).toFixed(2));
    const mk = (D.fixtures[c].find((m) => m.h === h && m.a === a) || {}).mk;
    return `<div class="in"><div><h3>Why this forecast</h3><div class="why">${why}</div>
      <p class="small" style="margin:12px 0 0">Bars to the right favour ${nm(h)}, to the left ${nm(a)}. The Elo gap includes 65 points for the home side.</p></div>
      <div><h3>The two teams</h3><table><thead><tr><th></th><th class="r">${nm(h)}</th><th class="r">${nm(a)}</th></tr></thead><tbody>
        <tr><td><a class="gl" href="#method" title="Strength score, see How it works">Elo rating</a></td><td class="r">${Math.round(ih.elo).toLocaleString("en-GB")}</td><td class="r">${Math.round(ia.elo).toLocaleString("en-GB")}</td></tr>
        <tr><td>Points, last 10</td><td class="r">${ih.pts10.toFixed(1)}</td><td class="r">${ia.pts10.toFixed(1)}</td></tr>
        <tr><td>Goals, last 10</td><td class="r">${ih.gf10.toFixed(1)}–${ih.ga10.toFixed(1)}</td><td class="r">${ia.gf10.toFixed(1)}–${ia.ga10.toFixed(1)}</td></tr>
        <tr><td>Squad value</td><td class="r">${sv(ih)}</td><td class="r">${sv(ia)}</td></tr></tbody></table>
      <p class="small" style="margin:14px 0 0">90% range for a ${nm(h)} win: ${pr(v[7])}–${pr(v[8])}%.${mk ? ` Bookmaker: ${pr(mk[2])} · ${pr(mk[1])} · ${pr(mk[0])}.` : ""}</p></div></div>`;
  }
  function dotGrid(rows, cols = 14, r = 8, gap = 10) {
    const n = rows.length, nr = Math.ceil(n / cols), W = Math.min(n, cols) * (2 * r + gap) - gap + 4, H = nr * (2 * r + gap) - gap + 4;
    const right = rows.filter(isRight).length;
    let s = `<svg class="dotgrid" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${right} of ${n} results called right">`;
    rows.forEach((x, i) => {
      const cx = 2 + r + (i % cols) * (2 * r + gap), cy = 2 + r + Math.floor(i / cols) * (2 * r + gap);
      s += isRight(x) ? `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${INK}"/>` : `<circle cx="${cx}" cy="${cy}" r="${r - 1}" fill="none" stroke="${INK}" stroke-width="2"/>`;
    });
    return s + "</svg>";
  }
  function rangeBar(lo, ex, hi, min, max, w = 170) {
    const X = (v) => 6 + (v - min) / (max - min) * (w - 12);
    return `<svg class="rng" width="${w}" height="22" viewBox="0 0 ${w} 22"><line x1="${X(lo)}" x2="${X(hi)}" y1="11" y2="11" stroke="${INK}" stroke-opacity=".4" stroke-width="2"/><circle cx="${X(ex)}" cy="11" r="5" fill="${INK}" stroke="${PAPER}" stroke-width="2"/></svg>`;
  }
  const tabs = (items, cur, attr) => `<div class="tabs" role="tablist">${items.map(([id, label]) => `<button role="tab" aria-selected="${id === cur}" data-${attr}="${id}">${esc(label)}</button>`).join("")}</div>`;
  const compTabs = (cur, attr) => tabs(COMPS.map((c) => [c, compName(c)]), cur, attr);
  const head = (eyebrow, h1, stand) => `<section class="head"><p class="eyebrow">${eyebrow}</p><h1>${h1}</h1>${stand ? `<p class="stand">${stand}</p>` : ""}</section>`;

  /* ------------------------------------------------------------------ PICKS */
  function pickRows() {
    const rows = LEAGUES.map((L) => {
      const lg = D.leagues[L];
      return { id: L, name: lg.name, sub: `after ${lg.rounds.played} of ${lg.rounds.total} rounds`, d: 0,
        list: Object.entries(lg.sim.summary).map(([t, v]) => ({ t, p: v[4] })).sort((a, b) => b.p - a.p) };
    });
    ["ucl", "uel"].forEach((c) => {
      const e = D.europe[c];
      rows.push({ id: c, name: e.name, sub: `${e.played} of ${e.total} league-phase matches played`, d: 1,
        list: e.teams.map((x) => ({ t: x.k, p: x.w })).sort((a, b) => b.p - a.p) });
    });
    return rows;
  }
  function pickRow(r) {
    const top = r.list[0], close = r.list.filter((x) => x.p >= top.p - 0.0105).length;
    const verdict = close >= 3 ? `${close} clubs within a point` : top.p >= 0.9 ? "near certain" : top.p >= 0.6 ? "clear favourite" : top.p >= 0.4 ? "favourite" : top.p >= 0.2 ? "open race" : "no clear favourite";
    const shown = r.list.slice(1, 4), me = r.list.findIndex((x) => x.t === FOCUS);
    const more = shown.map((x) => (x.t === FOCUS ? `<em>${nm(x.t)} ${pc(x.p, r.d)}</em>` : `${nm(x.t)} ${pc(x.p, r.d)}`));
    if (me > 3) more.push(`<em>${nm(FOCUS)} ${pc(r.list[me].p, r.d)}</em>`);
    return `<li><a class="pick" href="#standings/${r.id}"><div class="pk-comp"><b>${esc(r.name)}</b><small>${esc(r.sub)}</small></div>` +
      `<div class="pk-name">${top.t === FOCUS ? `<span class="me">${nm(top.t)}</span>` : nm(top.t)}</div>` +
      `<div class="pk-p"><b>${pc(top.p, r.d)}</b><span class="bar1"><i style="width:${(top.p * 100).toFixed(1)}%"></i></span><div class="pk-verdict">${verdict}</div></div>` +
      `<div class="pk-more">${more.join(" · ")}</div></a></li>`;
  }
  function lyonBlock() {
    const L = leagueOf(FOCUS), s = D.leagues[L].sim.summary[FOCUS], eu = euroOf(FOCUS), next = teamFixtures(FOCUS).slice(0, 3);
    const figs = [[s[0].toFixed(1), `points expected, 80% range ${Math.round(s[1])}–${Math.round(s[2])}`], [s[3].toFixed(1), "average final position"],
      [pc(s[5]), "chance of a Champions League place"], [pc(s[4]), "chance of the title"]];
    if (eu) figs.push([pc(eu.row.w, 1), `chance to win the ${compName(eu.comp)}`]);
    return `<section class="block"><h2>${nm(FOCUS)}: ${ordWord(Math.round(s[3]))} on average, ${pc(s[5])} for the Champions League.</h2>
      <div class="figs">${figs.map(([v, l]) => `<div class="fig"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("")}</div>
      <div class="cols" style="margin-top:48px"><div><h3>Next three matches</h3>
        <ul class="upnext">${next.map((m) => fxRow(m, { dated: true, expand: false })).join("") || '<li class="empty">No fixture left.</li>'}</ul>
        <a class="link cta" href="#team/${L}/${FOCUS}">Everything about ${nm(FOCUS)} →</a></div>
      <div><p class="chart-t">Where ${nm(FOCUS)} finishes, across 10,000 simulated seasons</p><div class="chart" data-chart="pos" data-l="${L}" data-t="${esc(FOCUS)}"></div>${posLegend(L, FOCUS)}</div></div></section>`;
  }
  function strip() {
    const P = D.prev, rows = P.rows.filter((r) => daysBetween(r[0], P.last) < 7).reverse(), W = P.summary["Last 7 days"];
    const right = rows.filter(isRight).length, first = rows[0][0];
    const better = W.ll < W.ll_book ? "The model did slightly better than the bookmaker" : "The bookmaker did slightly better";
    return `<section class="block"><h2>Between ${dDay(first)} and ${dDay(P.last)}, it called ${right} of ${rows.length} results right.</h2>
      <p class="lead">${better}: a log loss of ${W.ll_book.toFixed(3)} for the bookmaker against ${W.ll.toFixed(3)} for the model (lower is better).</p>
      ${dotGrid(rows)}<a class="link cta" href="#results">See every match →</a></section>`;
  }
  function pagePicks() {
    return head(`Updated ${fm(D.as_of, { day: "numeric", month: "long", year: "numeric" })} · results through ${dDay(D.data_through)}`, "Who wins this season?",
      "The model's pick in each competition. Click a row for the full table.") +
      `<ol class="picks">${pickRows().map(pickRow).join("")}</ol>` + lyonBlock() + strip();
  }

  /* ------------------------------------------------------------------ MATCHES */
  function pageMatches(comp) {
    comp = COMPS.includes(comp) ? comp : S.comp; S.comp = comp;
    const list = fixturesOf(comp), rounds = [...new Set(list.filter((m) => !m.late).map((m) => m.md))].sort((a, b) => a - b);
    const teams = [...new Set(list.flatMap((m) => [m.h, m.a]))].sort((a, b) => (D.display[a] || a).localeCompare(D.display[b] || b));
    if (!teams.includes(S.mteam)) S.mteam = "";
    let r = S.round[comp]; if (!rounds.includes(r)) r = S.round[comp] = rounds[0];
    const idx = rounds.indexOf(r), kind = isLeague(comp) ? "Round" : "Matchday";
    let items, sub;
    if (S.mteam) { items = list.filter((m) => m.h === S.mteam || m.a === S.mteam); sub = `${nm(S.mteam)}: ${items.length} matches to play`; }
    else {
      items = list.filter((m) => m.md === r || (m.late && idx === 0));
      const ds = items.filter((m) => !m.late).map((m) => m.d).sort();
      sub = `${kind} ${r}${ds.length ? ` · ${dDay(ds[0])}${ds[ds.length - 1] !== ds[0] ? " – " + dDay(ds[ds.length - 1]) : ""}` : ""}`;
    }
    const byDay = {};
    items.slice().sort(cmpFx).forEach((m) => (byDay[m.late ? "late" : m.d] ||= []).push(m));
    const days = Object.keys(byDay).sort((a, b) => (a === "late" ? -1 : b === "late" ? 1 : a.localeCompare(b)));
    const cls = items.some((m) => showTime(m) || m.late) ? "fxlist" : "fxlist notime";   // no empty column when no kick-off time is known
    const body = days.map((d) => `<div class="day">${d === "late" ? "Postponed, new date not published" : dHead(d)}<small>${S.mteam ? `${kind} ${byDay[d][0].md}` : `${byDay[d].length} ${byDay[d].length > 1 ? "matches" : "match"}`}</small></div><ul class="${cls}">${byDay[d].map((m) => fxRow(m, { comp })).join("")}</ul>`).join("") || '<p class="empty">No match left.</p>';
    const step = S.mteam ? "" : `<span class="step"><button data-step="-1" aria-label="Previous ${kind.toLowerCase()}" ${idx <= 0 ? "disabled" : ""}>‹</button><span>${sub}</span><button data-step="1" aria-label="Next ${kind.toLowerCase()}" ${idx >= rounds.length - 1 ? "disabled" : ""}>›</button></span>`;
    const note = !isLeague(comp) && D.europe[comp].missing ? `<p class="callout">${D.europe[comp].missing} matches played before ${dDay(D.as_of)} are not recorded in the data yet; they are left out of this list and simulated in the standings.</p>` : "";
    return head("Predictions for every match still to play", "Match predictions", `Left to right: home win, draw, away win. The bold side is the model's pick${isLeague(comp) ? "; click a match to see why" : ""}.`) +
      compTabs(comp, "comp") +
      `<div class="controls">${step}<label class="sel">Team <select id="sel-mteam"><option value="">All teams</option>${teams.map((t) => `<option value="${esc(t)}" ${t === S.mteam ? "selected" : ""}>${nm(t)}</option>`).join("")}</select></label>${S.mteam ? `<span class="small">${sub}</span>` : ""}</div>` +
      note + body;
  }

  /* ------------------------------------------------------------------ STANDINGS */
  function leagueTable(L) {
    const lg = D.leagues[L], sl = lg.slots;
    const rows = Object.entries(lg.sim.summary).map(([t, v]) => ({ t, exp: v[0], lo: v[1], hi: v[2], pos: v[3], title: v[4], cl: v[5], eu: v[8], rel: v[9], pts: v[10], n: v[11] })).sort((a, b) => a.pos - b.pos);
    const min = Math.min(...rows.map((r) => r.lo)) - 2, max = Math.max(...rows.map((r) => r.hi)) + 2;
    return `<div class="tw"><table><thead><tr><th class="r">#</th><th>Club</th><th class="r">P</th><th class="r">Pts</th><th colspan="2">Expected points, 80% range</th><th class="r">Title</th><th class="r">Champions League</th><th class="r">Europe</th><th class="r">Relegation</th></tr></thead><tbody>` +
      rows.map((r, i) => `<tr class="${r.t === FOCUS ? "me" : ""}"><td class="r">${i + 1}</td><td class="t">${nm(r.t)}</td><td class="r">${r.n}</td><td class="r">${r.pts}</td><td>${rangeBar(r.lo, r.exp, r.hi, min, max)}</td><td class="r"><b>${r.exp.toFixed(1)}</b></td><td class="r">${pt(r.title)}</td><td class="r">${pt(r.cl)}</td><td class="r">${pt(r.eu)}</td><td class="r">${pt(r.rel)}</td></tr>`).join("") +
      `</tbody></table></div><p class="small" style="margin-top:16px">Places assumed: Champions League = top ${sl.champions_league}, any European place = top ${sl.europe}, relegation = bottom ${sl.relegation}. Clubs are ordered by average final position over 10,000 simulated seasons.</p>`;
  }
  function euroTable(c) {
    const e = D.europe[c], rows = e.teams.slice().sort((a, b) => b.exp - a.exp || b.w - a.w);
    const missing = e.missing ? ` ${e.missing} matches already played are not in the data yet and are simulated.` : "";
    return `<div class="tw"><table><thead><tr><th class="r">#</th><th>Club</th><th class="r">P</th><th class="r">Pts</th><th class="r">Expected pts</th><th class="r">Top 8</th><th class="r">Top 24</th><th class="r">Final</th><th class="r">Win</th></tr></thead><tbody>` +
      rows.map((r, i) => `<tr class="${r.k === FOCUS ? "me" : ""}"><td class="r">${i + 1}</td><td class="t">${nm(r.k)}${r.assumed ? " †" : ""}</td><td class="r">${r.n}</td><td class="r">${r.pts}</td><td class="r">${r.exp.toFixed(1)}</td><td class="r">${pt(r.top8)}</td><td class="r">${pt(r.top24)}</td><td class="r">${pt(r.f)}</td><td class="r"><b>${pw(r.w)}</b></td></tr>`).join("") +
      `</tbody></table></div><p class="small" style="margin-top:16px">Top 8 go straight to the round of 16, ranks 9 to 24 play the knockout play-offs. Expected points are for the end of the league phase.${missing}${e.n_assumed ? ` † Club missing from the squad-value data: strength assumed low (${e.n_assumed} clubs).` : ""}</p>`;
  }
  function pageStandings(comp) {
    comp = COMPS.includes(comp) ? comp : "F1";
    const title = isLeague(comp) ? `${compName(comp)}: projected final table` : `${compName(comp)}: projected league phase`;
    return head("Where each club is expected to finish", title) + compTabs(comp, "stand") + (isLeague(comp) ? leagueTable(comp) : euroTable(comp)) +
      `<a class="link cta" href="#matches/${comp}">Predictions for the next matches →</a>`;
  }

  /* ------------------------------------------------------------------ TEAM */
  function pageTeam(L, team) {
    L = LEAGUES.includes(L) ? L : "F1";
    const lg = D.leagues[L]; if (!lg.teams.includes(team)) team = L === leagueOf(FOCUS) ? FOCUS : lg.teams[0];
    const s = lg.sim.summary[team], info = lg.info[team], eu = euroOf(team), next = teamFixtures(team).slice(0, 3);
    const figs = [[s[0].toFixed(1), `points expected, 80% range ${Math.round(s[1])}–${Math.round(s[2])}`], [s[3].toFixed(1), "average final position"],
      [pc(s[4]), "title"], [pc(s[5]), `Champions League places (top ${lg.slots.champions_league})`], [s[9] < 0.005 ? "&lt;1%" : pc(s[9]), "relegation"]];
    if (eu) figs.push([pc(eu.row.w, 1), `to win the ${compName(eu.comp)}`]);
    const sv = info.sv === null ? "" : ` · squad value ×${Math.exp(info.sv).toFixed(2)} the league average`;
    const teamsSorted = lg.teams.slice().sort((a, b) => (D.display[a] || a).localeCompare(D.display[b] || b));
    return head(`${esc(lg.name)} · after ${s[11]} rounds`, nm(team),
      `${s[10]} points · <a class="gl" href="#method">Elo</a> ${Math.round(info.elo).toLocaleString("en-GB")} (${ordNum(info.elo_rank)} of ${lg.teams.length})${sv}`) +
      `<div class="controls"><label class="sel">League <select id="sel-tl">${LEAGUES.map((l) => `<option value="${l}" ${l === L ? "selected" : ""}>${esc(D.leagues[l].name)}</option>`).join("")}</select></label>
       <label class="sel">Team <select id="sel-tt">${teamsSorted.map((t) => `<option value="${esc(t)}" ${t === team ? "selected" : ""}>${nm(t)}</option>`).join("")}</select></label></div>` +
      `<div class="figs">${figs.map(([v, l]) => `<div class="fig"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("")}</div>` +
      `<div class="cols" style="margin-top:56px"><div><h3>Next matches</h3><ul class="upnext">${next.map((m) => fxRow(m, { dated: true, focus: team, expand: false })).join("") || '<li class="empty">No fixture left.</li>'}</ul>
        <a class="link cta" href="#matches/${L}">All ${esc(lg.name)} predictions →</a></div>
       <div><p class="chart-t">Where ${nm(team)} finishes, across 10,000 simulated seasons</p><div class="chart" data-chart="pos" data-l="${L}" data-t="${esc(team)}"></div>${posLegend(L, team)}</div></div>` +
      `<section class="block"><h2>Strength since 2016</h2><p class="lead">Elo rating of ${nm(team)} (red) against the rest of the ${esc(lg.name)}.</p><div class="chart" data-chart="elo" data-l="${L}" data-t="${esc(team)}"></div></section>`;
  }

  /* ------------------------------------------------------------------ RESULTS */
  function pageResults() {
    const P = D.prev, names = ["Away win", "Draw", "Home win"];
    const sum = Object.entries(P.summary).map(([k, v]) => `<div><div class="l">${esc(k.replace(" (2023 → Oct 2026)", ""))}</div><div class="v">${pr(v.acc)}%</div><div class="s">called right, ${v.n} matches<br>Log loss: model ${v.ll.toFixed(3)} · bookmaker ${v.ll_book.toFixed(3)} · no skill ${v.ll_naive.toFixed(3)}</div></div>`).join("");
    const cal = `<div class="tw"><table><thead><tr><th>Top pick given</th><th class="r">Matches</th><th class="r">Average forecast</th><th class="r">Came true</th></tr></thead><tbody>${P.calibration.map((b) => `<tr><td>${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}%</td><td class="r">${b.n}</td><td class="r">${pc(b.said)}</td><td class="r"><b>${pc(b.came_true)}</b></td></tr>`).join("")}</tbody></table></div>`;
    const rows = P.rows.filter((r) => !S.rlg || r[1] === S.rlg).slice(0, 80);
    const tbl = `<div class="tw"><table><thead><tr><th>Date</th><th>Match</th><th class="r">Score</th><th class="r">Home · draw · away</th><th>Top pick</th><th>Result</th></tr></thead><tbody>` + rows.map((r) => {
      const best = r[7].indexOf(Math.max(...r[7])), ok = best === r[6], mine = r[2] === FOCUS || r[3] === FOCUS;
      return `<tr class="${mine ? "me" : ""}"><td>${dDay(r[0])}</td><td class="t">${nm(r[2])} – ${nm(r[3])}</td><td class="r"><b>${r[4]}–${r[5]}</b></td><td class="r">${pr(r[7][2])} · ${pr(r[7][1])} · ${pr(r[7][0])}</td><td>${names[best]} ${pr(r[7][best])}%</td><td class="${ok ? "ok-t" : "no-t"}"><span class="dot ${ok ? "ok" : "no"}"></span>${ok ? "Right" : "Missed"}</td></tr>`;
    }).join("") + `</tbody></table></div>`;
    return head(`Matches played up to ${dDay(P.last)}`, "Was the model right?", "Every recent match, with the probabilities the model gave before kick-off.") +
      `<p class="callout">These forecasts were recomputed after the fact, with the model fitted only on earlier seasons and variables built only from earlier matches. They were not published in advance. "Right" means the most likely outcome happened: a 45% pick is wrong more often than right, so log loss and the calibration table are the fair scores.</p>` +
      `<div class="sumrow">${sum}</div>` +
      `<section class="block"><h2>Do the percentages mean what they say?</h2><p class="lead">When the model gave 60–70% to its top pick, it happened ${pc(P.calibration[3].came_true)} of the time, over ${P.n_total.toLocaleString("en-GB")} matches since ${fm(P.from, { month: "long", year: "numeric" })}.</p>${cal}</section>` +
      `<section class="block"><h2>Match by match</h2><div class="controls"><label class="sel">League <select id="sel-rlg"><option value="">All leagues</option>${LEAGUES.map((l) => `<option value="${l}" ${l === S.rlg ? "selected" : ""}>${esc(D.leagues[l].name)}</option>`).join("")}</select></label></div>${tbl}</section>`;
  }

  /* ------------------------------------------------------------------ METHOD */
  function eloSection() {
    const E = (rh, ra) => 1 / (1 + Math.pow(10, -(rh - ra + 65) / 400)), r0 = (x) => Math.round(x).toLocaleString("en-GB");
    const m = teamFixtures(FOCUS).find((x) => isLeague(x.comp)), lg = m && D.leagues[m.comp];
    let ex = "";
    if (m) {
      const rh = Math.round(lg.info[m.h].elo), ra = Math.round(lg.info[m.a].elo), e = E(rh, ra);
      ex = `<p><b>Example.</b> ${nm(m.h)} (${r0(rh)}) host ${nm(m.a)} (${r0(ra)}) on ${fm(m.d, { day: "numeric", month: "long" })}. The gap is ${r0(rh - ra)} points; with the 65 points for playing at home it becomes ${r0(rh - ra + 65)}, which gives ${nm(m.h)} an expected score of ${pc(e)}. A win barely moves its rating, because it was expected; a defeat would cost it a lot.</p>`;
    }
    return `<section class="block"><div class="prose"><h2>Elo: one number for team strength</h2>
        <p>Elo is a strength score borrowed from chess. Every club has one. It rises after a win and falls after a defeat, by a lot when the opponent was stronger than expected and by little when it was weaker. A club starts a league at 1,500; a promoted club starts 75 points lower. Each league has its own scale, so a Premier League 1,600 and a Ligue 1 1,600 cannot be compared.</p>
        <p>Before a match, the gap between the two ratings gives an expected score for the home side (a win counts 1, a draw one half, a defeat 0):</p>
        <div class="formula">E = 1 / (1 + 10<sup>−(R<sub>home</sub> − R<sub>away</sub> + 65) / 400</sup>)</div>
        <p>The <b>+65</b> is the home advantage: the home side is treated as 65 rating points stronger. Against an equal opponent that is an expected score of about 59%; in the data, home sides collect 57% of the points on offer.</p>
        <p>After the match, the rating moves by the surprise, the gap between the actual result S and the expected score E:</p>
        <div class="formula">R′ = R + 20 × (1 + ln(1 + |goal difference|)) × (S − E)</div>
        <p>The 20 sets how fast ratings react, and the logarithm makes a 4–0 count more than a 1–0 without letting it dominate. Between seasons every rating moves a quarter of the way back to the league average. The rating used for a match is always the one from before it was played.</p>${ex}
        <p>Elo is the input the model leans on most, ahead of squad market value. Everything else adds little once the two are known.</p></div></section>`;
  }
  function pageMethod() {
    const A = D.arena.final, names = [["LogReg (final)", "Logistic regression (chosen)"], ["Gradient boosting", "Gradient boosting"], ["MLP deep (5 seeds)", "Neural network"], ["Ensemble LogReg+GB+MLP", "Ensemble of the three"], ["Elo (LogReg)", "Elo"], ["Class frequency (naive)", "Ignoring team quality"], ["Bookmaker Bet365 (reference)", "Bookmaker (Bet365)"]];
    const tbl = `<div class="tw"><table><thead><tr><th>Forecaster</th><th class="r">Log loss</th><th class="r">95% interval</th><th class="r">Right</th></tr></thead><tbody>${names.map(([k, l]) => `<tr class="${k === "LogReg (final)" ? "me" : ""}"><td class="t">${l}</td><td class="r"><b>${A[k].log_loss.toFixed(3)}</b></td><td class="r">${A[k].lo.toFixed(3)}–${A[k].hi.toFixed(3)}</td><td class="r">${pc(A[k].accuracy, 1)}</td></tr>`).join("")}</tbody></table></div>`;
    const lr = A["LogReg (final)"].log_loss, bk = A["Bookmaker Bet365 (reference)"].log_loss;
    return head("What is running behind the numbers", "How it works") +
      `<section class="block" style="padding-top:48px"><div class="prose"><h2>One model: a logistic regression</h2>
        <p>Each match gets three probabilities, home win, draw and away win, from a multinomial logistic regression on 100 inputs: Elo rating, squad market value, recent results and shots, schedule. It is refitted every season on all past matches.</p>
        <div class="formula">P(k | x) = exp(wₖ · x + bₖ) / Σⱼ exp(wⱼ · x + bⱼ)</div>
        <p>Random forests, gradient boosting, support-vector machines and neural networks were tested too. None beat it significantly, so the simplest and most explainable model stays.</p></div></section>` +
      eloSection() +
      `<section class="block"><h2>How good is it?</h2><p class="lead">Since ${fm(D.prev.from, { month: "long", year: "numeric" })}, on ${D.prev.n_total.toLocaleString("en-GB")} matches, the chosen model scores ${lr.toFixed(3)} (log loss, lower is better). The bookmaker scores ${bk.toFixed(3)}: better, and adding the model to its prices adds nothing.</p>
        <p class="chart-t">Average log loss over the test matches, as they accumulate</p><div class="chart ar" data-chart="arena"></div>
        <div class="legend"><span><i style="background:${ACCENT}"></i>Logistic regression (chosen)</span><span><i style="background:${MID}"></i>Boosting, neural network, ensemble</span><span><i style="background:${INK}"></i>Bookmaker</span><span><i style="background:${GRAY}"></i>Elo, and ignoring team quality</span></div>${tbl}</section>` +
      `<section class="block"><div class="prose"><h2>From matches to titles</h2>
        <p>The season is replayed 10,000 times: each remaining match draws its result from the model's probabilities, after a random shock to every team's strength of 60 Elo points, a size chosen by a backtest on 2023 to 2025. Counting the replays gives the title, qualification and relegation odds.</p>
        <p>The Champions League and the Europa League use a second model fitted on 3,085 past European matches, from squad value and domestic form, then replay the league phase, the play-offs and the knockout rounds 20,000 times.</p></div></section>` +
      `<section class="block"><div class="prose"><h2>Limits</h2><ul class="list">
        <li>No injuries, line-ups or coaching news: the likely reason the bookmaker is better.</li>
        <li>Team strength is frozen at ${dDay(D.as_of)}; forecasts move after every round.</li>
        <li>Calendars and dates can change. A match still unplayed after its date is flagged as postponed.</li>
        <li>Qualification places are assumptions, listed under each league table.</li>
        <li>The European model is tested match by match, not on whole tournaments. Clubs missing from the squad-value data are assumed weak.</li></ul>
        <a class="link" href="https://github.com/lacostetheo/ML-PREDICTION-OLYMPIQUE-LYONNAIS">Code, data sources and the full paper on GitHub →</a></div></section>`;
  }

  /* ------------------------------------------------------------------ charts (drawn at the real width, so text stays 14 px+) */
  const svg = (w, h, inner, label) => `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}">${inner}</svg>`;
  function niceStep(max) {
    for (const s of [0.05, 0.1, 0.2, 0.25, 0.5]) { const top = Math.ceil(max / s - 1e-9) * s; if (top / s <= 4) return [s, top]; }
    return [0.5, 1];
  }
  function posChart(el) {
    const L = el.dataset.l, team = el.dataset.t, lg = D.leagues[L], probs = lg.sim.pos_probs[team], n = probs.length;
    const hot = lg.slots.champions_league, rel = lg.slots.relegation;
    const W = Math.max(280, Math.floor(el.clientWidth)), H = W < 480 ? 290 : 330, ml = 44, mr = 6, mt = 64, mb = 34;
    const pw = W - ml - mr, ph = H - mt - mb, slot = pw / n, bw = Math.min(24, slot * 0.62), max = Math.max(...probs);
    const [step, ymax] = niceStep(max * 1.15), Y = (v) => mt + ph * (1 - v / ymax), y0 = Y(0);
    let s = "";
    for (let g = 0; g <= ymax + 1e-9; g += step) s += `<line x1="${ml}" x2="${W - mr}" y1="${Y(g)}" y2="${Y(g)}" stroke="${RULE}"/><text x="${ml - 8}" y="${Y(g) + 5}" text-anchor="end" font-size="14" fill="${INK3}">${Math.round(g * 100)}%</text>`;
    const bar = (x, y, w, r = 4) => { r = Math.min(r, w / 2, y0 - y); return `M${x},${y0} L${x},${y + r} Q${x},${y} ${x + r},${y} L${x + w - r},${y} Q${x + w},${y} ${x + w},${y + r} L${x + w},${y0} Z`; };
    probs.forEach((p, i) => {
      const x = ml + i * slot + (slot - bw) / 2, col = i < hot ? ACCENT : i >= n - rel ? MID : GRAY;
      if (p > 0.0005) s += `<path d="${bar(x, Y(p), bw)}" fill="${col}"><title>${ordNum(i + 1)}: ${(p * 100).toFixed(1)}%</title></path>`;
      if (slot >= 22 || i % 2 === 0) s += `<text x="${x + bw / 2}" y="${H - mb + 24}" text-anchor="middle" font-size="14" fill="${INK3}">${i + 1}</text>`;
    });
    const im = probs.indexOf(max);
    s += `<text x="${ml + im * slot + slot / 2}" y="${Y(max) - 8}" text-anchor="middle" font-size="15" font-weight="600" fill="${INK}" stroke="${PAPER}" stroke-width="4" paint-order="stroke">${pr(max)}%</text>`;
    const bracket = (i0, i1, label) => {                // sits above the top gridline, so it never crosses a bar, a label or a line
      const x1 = ml + i0 * slot + (slot - bw) / 2, x2 = ml + i1 * slot + (slot - bw) / 2 + bw, yb = mt - 24;
      return `<path d="M${x1},${yb + 8} L${x1},${yb} L${x2},${yb} L${x2},${yb + 8}" fill="none" stroke="${ACCENT}" stroke-width="2"/><text x="${x1}" y="${yb - 10}" font-size="15" font-weight="600" fill="${INK}">${label}</text>`;
    };
    const pHot = probs.slice(0, hot).reduce((a, b) => a + b, 0);
    s += bracket(0, hot - 1, `Champions League: ${pc(pHot)}`);
    el.innerHTML = svg(W, H, s, `Distribution of the final position of ${D.display[team] || team}`);
  }
  function posLegend(L, team) {
    const lg = D.leagues[L], p = lg.sim.pos_probs[team], n = p.length, sum = (a, b) => p.slice(a, b).reduce((x, y) => x + y, 0);
    return `<div class="legend"><span><i style="background:${ACCENT}"></i>Champions League places (top ${lg.slots.champions_league}): ${pc(sum(0, lg.slots.champions_league))}</span><span><i style="background:${MID}"></i>Relegation places (bottom ${lg.slots.relegation}): ${sum(n - lg.slots.relegation, n) < 0.005 ? "under 1%" : pc(sum(n - lg.slots.relegation, n))}</span></div>`;
  }
  function eloChart(el) {
    const L = el.dataset.l, team = el.dataset.t, lg = D.leagues[L], months = D.elo_months, start = months.indexOf("2016-08");
    const ser = (t) => lg.info[t].elo_series.slice(start), all = lg.teams.flatMap((t) => ser(t).filter((v) => v !== null));
    const lo = Math.floor(Math.min(...all) / 100) * 100, hi = Math.ceil(Math.max(...all) / 100) * 100;
    const W = Math.max(280, Math.floor(el.clientWidth)), H = W < 480 ? 280 : 340, ml = 52, mr = 14, mt = 34, mb = 34, pw = W - ml - mr, ph = H - mt - mb, N = months.length - start;
    const X = (i) => ml + (i / (N - 1)) * pw, Y = (v) => mt + ph * (1 - (v - lo) / (hi - lo));
    let s = "";
    for (let g = lo; g <= hi; g += 100) s += `<line x1="${ml}" x2="${W - mr}" y1="${Y(g)}" y2="${Y(g)}" stroke="${RULE}"/><text x="${ml - 8}" y="${Y(g) + 5}" text-anchor="end" font-size="14" fill="${INK3}">${g.toLocaleString("en-GB")}</text>`;
    months.slice(start).forEach((m, i) => { if (m.endsWith("-01") && (+m.slice(0, 4)) % (W < 480 ? 3 : 2) === 0) s += `<text x="${X(i)}" y="${H - 8}" text-anchor="middle" font-size="14" fill="${INK3}">${m.slice(0, 4)}</text>`; });
    const path = (vals) => { let d = "", on = false; vals.forEach((v, i) => { if (v === null) { on = false; return; } d += `${on ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)} `; on = true; }); return d; };
    lg.teams.filter((t) => t !== team).forEach((t) => (s += `<path d="${path(ser(t))}" fill="none" stroke="${GRAY}" stroke-width="1" stroke-linejoin="round"><title>${D.display[t] || t}</title></path>`));
    const mine = ser(team), last = mine.length - 1 - [...mine].reverse().findIndex((v) => v !== null);
    s += `<path d="${path(mine)}" fill="none" stroke="${ACCENT}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/><circle cx="${X(last)}" cy="${Y(mine[last])}" r="5" fill="${ACCENT}" stroke="${PAPER}" stroke-width="2"/>`;
    s += `<text x="${X(last)}" y="${Y(mine[last]) - 14}" text-anchor="end" font-size="15" font-weight="600" fill="${INK}" stroke="${PAPER}" stroke-width="4" paint-order="stroke">${D.display[team] || team} ${Math.round(lg.info[team].elo).toLocaleString("en-GB")}</text>`;
    el.innerHTML = svg(W, H, s, `Elo rating of ${D.display[team] || team} since 2016`);
  }
  function arenaChart(el) {
    const A = D.arena, i0 = A.n.findIndex((n) => n >= 450), xs = A.n.slice(i0), W = Math.max(280, Math.floor(el.clientWidth));
    const wide = W >= 700, H = W < 480 ? 300 : 360, ml = 52, mr = wide ? 170 : 12, mt = 16, mb = 34, pw = W - ml - mr, ph = H - mt - mb;
    const cur = (k) => A.curves[k].slice(i0), vals = Object.keys(A.curves).flatMap(cur);
    const lo = Math.floor(Math.min(...vals) * 20) / 20, hi = Math.ceil(Math.max(...vals) * 20) / 20;
    const X = (i) => ml + (i / (xs.length - 1)) * pw, Y = (v) => mt + ph * (1 - (v - lo) / (hi - lo));
    let s = "";
    for (let g = lo; g <= hi + 1e-9; g += 0.05) s += `<line x1="${ml}" x2="${W - mr}" y1="${Y(g)}" y2="${Y(g)}" stroke="${RULE}"/><text x="${ml - 8}" y="${Y(g) + 5}" text-anchor="end" font-size="14" fill="${INK3}">${g.toFixed(2)}</text>`;
    A.dates.slice(i0).forEach((d, i, arr) => { if (d.slice(5, 7) === "01" && (i === 0 || arr[i - 1].slice(5, 7) !== "01")) s += `<text x="${X(i)}" y="${H - 8}" text-anchor="middle" font-size="14" fill="${INK3}">${d.slice(0, 4)}</text>`; });
    const line = (k, col, w) => `<path d="${cur(k).map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ")}" fill="none" stroke="${col}" stroke-width="${w}" stroke-linejoin="round"><title>${k}</title></path>`;
    s += line("Class frequency (naive)", GRAY, 1.5) + line("Elo (LogReg)", MID, 1.5) + line("Gradient boosting", MID, 1.5) + line("MLP deep (5 seeds)", MID, 1.5) + line("Ensemble LogReg+GB+MLP", MID, 1.5) + line("LogReg (final)", ACCENT, 2.5) + line("Bookmaker Bet365 (reference)", INK, 2.5);
    if (wide) {
      const end = (k) => cur(k)[xs.length - 1];
      const L = [["Bookmaker Bet365 (reference)", "Bookmaker"], ["LogReg (final)", "Our models"], ["Elo (LogReg)", "Elo"], ["Class frequency (naive)", "No skill"]]
        .map(([k, t]) => ({ t, v: end(k), y: Y(end(k)) })).sort((p, q) => p.y - q.y);
      for (let i = 1; i < L.length; i++) L[i].y = Math.max(L[i].y, L[i - 1].y + 20);          // keep end labels 20 px apart
      s += L.map((o) => `<text x="${W - mr + 12}" y="${o.y + 5}" font-size="15" fill="${INK}">${o.t} <tspan font-weight="600">${o.v.toFixed(3)}</tspan></text>`).join("");
    }
    el.innerHTML = svg(W, H, s, "Cumulative log loss of the forecasters over the test matches");
  }
  function drawCharts() { $$(".chart[data-chart]").forEach((el) => ({ pos: posChart, elo: eloChart, arena: arenaChart }[el.dataset.chart](el))); }

  /* ------------------------------------------------------------------ router */
  const TITLES = { picks: "Picks", matches: "Match predictions", standings: "Projected tables", team: "Team", results: "Was the model right?", method: "How it works" };
  function route() {
    const [p0, a, b] = location.hash.slice(1).split("/").map(decodeURIComponent);
    const page = TITLES[p0] ? p0 : "picks";
    const html = { picks: () => pagePicks(), matches: () => pageMatches(a), standings: () => pageStandings(a), team: () => pageTeam(a, b), results: () => pageResults(), method: () => pageMethod() }[page]();
    const same = S.page === page; S.page = page;
    $("#app").innerHTML = html;
    $$("#nav a").forEach((x) => (x.getAttribute("data-p") === page ? x.setAttribute("aria-current", "page") : x.removeAttribute("aria-current")));
    document.title = `${TITLES[page]} · OL·LAB`;
    drawCharts();
    if (!same) window.scrollTo(0, 0);
  }
  document.addEventListener("click", (e) => {
    const k = e.target.closest(".fx[data-k]");
    if (k) {
      const open = k.getAttribute("aria-expanded") === "true", box = k.nextElementSibling;
      if (!open && !box.innerHTML) box.innerHTML = detail(k.dataset.k);
      k.setAttribute("aria-expanded", String(!open)); box.hidden = open; return;
    }
    const c = e.target.closest("[data-comp]"); if (c) { location.hash = `#matches/${c.dataset.comp}`; return; }
    const t = e.target.closest("[data-stand]"); if (t) { location.hash = `#standings/${t.dataset.stand}`; return; }
    const st = e.target.closest("[data-step]");
    if (st && !st.disabled) {
      const list = fixturesOf(S.comp), rounds = [...new Set(list.filter((m) => !m.late).map((m) => m.md))].sort((x, y) => x - y);
      S.round[S.comp] = rounds[Math.max(0, Math.min(rounds.length - 1, rounds.indexOf(S.round[S.comp]) + Number(st.dataset.step)))]; route();
    }
  });
  document.addEventListener("change", (e) => {
    if (e.target.id === "sel-mteam") { S.mteam = e.target.value; route(); }
    if (e.target.id === "sel-rlg") { S.rlg = e.target.value; route(); }
    if (e.target.id === "sel-tl") { const L = e.target.value; location.hash = `#team/${L}/${encodeURIComponent(L === leagueOf(FOCUS) ? FOCUS : D.leagues[L].teams[0])}`; }
    if (e.target.id === "sel-tt") location.hash = `#team/${$("#sel-tl").value}/${encodeURIComponent(e.target.value)}`;
  });
  let timer; window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(drawCharts, 150); });
  window.addEventListener("hashchange", route);
  $("#upd").textContent = `Updated ${fm(D.as_of, { day: "numeric", month: "short", year: "numeric" })}`;
  $("#refresh").textContent = `Last updated ${fm(D.as_of, { weekday: "long", day: "numeric", month: "long", year: "numeric" })}, with results through ${dDay(D.data_through)}. The numbers are recomputed every Monday.`;
  route();
})();
