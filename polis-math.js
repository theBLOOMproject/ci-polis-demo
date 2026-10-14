// Polis opinion math, ported to plain JS.
//
// Follows the pipeline in compdemocracy/polis (math/):
//   1. Vote matrix: participants × comments, agree = +1, disagree = -1, pass = 0, unseen = missing.
//      (Polis's own HTTP API encodes agree as -1; this module uses the intuitive sign.)
//   2. PCA to 2 components. Missing votes are imputed with the comment's mean vote.
//   3. Sparsity-aware projection: a participant's projection is scaled by sqrt(nComments / nVotes)
//      so people who voted on few comments aren't all pulled into the centre.
//   4. Only participants with at least min(7, nComments) votes are clustered ("in conversation").
//   5. k-means for k = 2..5, picking k by silhouette score, with some stickiness to the previous k.
//   6. Representativeness (repness): for each group and comment, how much more the group agrees
//      (or disagrees) than everyone else, gated by one-tailed 90% proportion tests.
//   7. Group-aware consensus: the product across groups of each group's smoothed agree rate.
//
// Simplification: Polis first compresses participants into ~100 "base clusters" and clusters those.
// At demo scale (a few hundred people) clustering participants directly gives the same result.
(function (root) {
  "use strict";

  const Z90 = 1.2816;        // one-tailed 90% critical value used by Polis's significance tests
  const MAX_K = 5;
  const MIN_VOTES = 7;
  const MAX_REPNESS = 5;
  const K_STICKINESS = 0.02; // stand-in for Polis's k-smoothing buffer: keep the old k unless clearly beaten

  function rngFrom(a) {
    return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }

  // Polis's prop-test: z-score that a proportion exceeds 0.5, with +1 pseudocounts.
  function propTest(succ, n) { succ += 1; n += 1; return 2 * Math.sqrt(n) * (succ / n - 0.5); }

  // Polis's two-prop-test: z-score that proportion 1 exceeds proportion 2, with +1 pseudocounts.
  function twoPropTest(s1, n1, s2, n2) {
    s1 += 1; n1 += 1; s2 += 1; n2 += 1;
    const p1 = s1 / n1, p2 = s2 / n2, ph = (s1 + s2) / (n1 + n2);
    const d = Math.sqrt(ph * (1 - ph) * (1 / n1 + 1 / n2));
    return d ? (p1 - p2) / d : 0;
  }

  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const dist2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

  function orthonormalize(v, basis) {
    v = v.slice();
    for (const b of basis) { const d = dot(v, b); for (let j = 0; j < v.length; j++) v[j] -= d * b[j]; }
    const n = Math.sqrt(dot(v, v)) || 1;
    return v.map(x => x / n);
  }

  // ---------- PCA (power iteration with deflation) ----------
  function pca(rows, m, prevComps) {
    const sum = new Float64Array(m), cnt = new Float64Array(m);
    for (const r of rows) for (let j = 0; j < m; j++) if (r[j] !== undefined) { sum[j] += r[j]; cnt[j]++; }
    const center = Array.from(sum, (s, j) => cnt[j] ? s / cnt[j] : 0);
    // Mean imputation then centering leaves missing cells at exactly 0.
    const C = rows.map(r => Array.from({ length: m }, (_, j) => r[j] === undefined ? 0 : r[j] - center[j]));

    const comps = [];
    for (let c = 0; c < 2; c++) {
      const rng = rngFrom(17 + c);
      const warm = prevComps && prevComps[c] && dot(prevComps[c], prevComps[c]) > 0;
      let v = orthonormalize(warm ? prevComps[c] : Array.from({ length: m }, () => rng() - 0.5), comps);
      for (let it = 0; it < 500; it++) {
        const u = new Array(m).fill(0);
        for (const row of C) { const w = dot(row, v); if (w) for (let j = 0; j < m; j++) u[j] += row[j] * w; }
        const next = orthonormalize(u, comps);
        let diff = 0; for (let j = 0; j < m; j++) diff += Math.abs(next[j] - v[j]);
        v = next;
        if (diff < 1e-10) break;
      }
      // Eigenvectors have arbitrary sign; align with the previous run so the map doesn't flip.
      const ref = warm ? prevComps[c] : null;
      if ((ref ? dot(v, ref) : v.reduce((a, b) => a + b, 0)) < 0) v = v.map(x => -x);
      comps.push(v);
    }
    return { center, comps };
  }

  function project(row, pc, m) {
    let x = 0, y = 0, n = 0;
    for (let j = 0; j < m; j++) {
      if (row[j] === undefined) continue;
      n++;
      const d = row[j] - pc.center[j];
      x += d * pc.comps[0][j]; y += d * pc.comps[1][j];
    }
    if (!n) return null;
    const s = Math.sqrt(m / n);
    return [x * s, y * s];
  }

  // ---------- k-means ----------
  function kmeans(pts, k, seed) {
    const rng = rngFrom(seed);
    const cents = [pts[Math.floor(rng() * pts.length)].slice()];
    while (cents.length < k) { // k-means++ seeding
      const d = pts.map(p => Math.min(...cents.map(c => dist2(p, c))));
      const tot = d.reduce((a, b) => a + b, 0);
      if (!tot) return null;
      let r = rng() * tot, i = 0;
      while ((r -= d[i]) > 0 && i < pts.length - 1) i++;
      cents.push(pts[i].slice());
    }
    const labels = new Array(pts.length).fill(-1);
    for (let it = 0; it < 100; it++) {
      let changed = false;
      pts.forEach((p, i) => {
        let best = 0, bd = Infinity;
        cents.forEach((c, ci) => { const d = dist2(p, c); if (d < bd) { bd = d; best = ci; } });
        if (labels[i] !== best) { labels[i] = best; changed = true; }
      });
      for (let ci = 0; ci < k; ci++) {
        let sx = 0, sy = 0, n = 0;
        pts.forEach((p, i) => { if (labels[i] === ci) { sx += p[0]; sy += p[1]; n++; } });
        if (n) cents[ci] = [sx / n, sy / n];
      }
      if (!changed) break;
    }
    const sizes = new Array(k).fill(0); labels.forEach(l => sizes[l]++);
    if (sizes.some(s => !s)) return null;
    return { labels, cents, sizes, inertia: pts.reduce((s, p, i) => s + dist2(p, cents[labels[i]]), 0) };
  }

  function bestKmeans(pts, k) {
    let best = null;
    for (let s = 0; s < 8; s++) { const r = kmeans(pts, k, 1000 * k + s); if (r && (!best || r.inertia < best.inertia)) best = r; }
    return best;
  }

  function silhouette(pts, labels, k) {
    let total = 0;
    for (let i = 0; i < pts.length; i++) {
      const sum = new Array(k).fill(0), cnt = new Array(k).fill(0);
      for (let j = 0; j < pts.length; j++) {
        if (i === j) continue;
        sum[labels[j]] += Math.sqrt(dist2(pts[i], pts[j])); cnt[labels[j]]++;
      }
      const own = labels[i];
      if (!cnt[own]) continue; // singleton cluster scores 0
      const a = sum[own] / cnt[own];
      let b = Infinity;
      for (let c = 0; c < k; c++) if (c !== own && cnt[c]) b = Math.min(b, sum[c] / cnt[c]);
      total += (b - a) / Math.max(a, b) || 0;
    }
    return total / pts.length;
  }

  // Keep group letters stable between runs by matching new centroids to the previous ones.
  function labelMap(cents, sizes, prevCents) {
    const k = cents.length, map = new Array(k).fill(-1), used = new Set();
    if (prevCents) {
      const pairs = [];
      cents.forEach((c, i) => prevCents.forEach((p, j) => { if (j < k) pairs.push([dist2(c, p), i, j]); }));
      pairs.sort((a, b) => a[0] - b[0]);
      for (const [, i, j] of pairs) if (map[i] < 0 && !used.has(j)) { map[i] = j; used.add(j); }
    }
    let next = 0;
    for (const i of [...Array(k).keys()].filter(i => map[i] < 0).sort((a, b) => sizes[b] - sizes[a])) {
      while (used.has(next)) next++;
      map[i] = next; used.add(next);
    }
    return map;
  }

  // ---------- Main entry ----------
  // participants: ids; comments: ids; vote(pid, tid) → 1 | -1 | 0 | undefined.
  // prev: the `memo` returned by the previous call, for stable signs, k and group letters.
  function compute({ participants, comments, vote, prev }) {
    const m = comments.length;
    const rows = new Map();
    for (const pid of participants) {
      const row = new Array(m); let n = 0;
      comments.forEach((tid, j) => { const v = vote(pid, tid); if (v !== undefined) { row[j] = v; n++; } });
      if (n) rows.set(pid, { row, n });
    }
    const minVotes = Math.min(MIN_VOTES, m);
    const inConv = [...rows.keys()].filter(pid => rows.get(pid).n >= minVotes);
    const out = { k: 0, groups: [], groupOf: {}, proj: {}, inConv: new Set(inConv), minVotes, voteCount: {}, repness: [], consensus: {}, counts: {}, memo: null };
    for (const [pid, { n }] of rows) out.voteCount[pid] = n;
    if (inConv.length < 3 || m < 2) return out;

    const prevComps = prev && prev.comps ? [0, 1].map(c => comments.map(tid => (prev.comps[tid] || [0, 0])[c])) : null;
    const pc = pca(inConv.map(pid => rows.get(pid).row), m, prevComps);
    for (const [pid, { row }] of rows) out.proj[pid] = project(row, pc, m);

    let centroids = [];
    if (inConv.length >= 4) {
      const pts = inConv.map(pid => out.proj[pid]);
      const cands = [];
      for (let k = 2; k <= Math.min(MAX_K, inConv.length - 1); k++) {
        const km = bestKmeans(pts, k);
        if (km) cands.push({ k, km, sil: silhouette(pts, km.labels, k) });
      }
      if (cands.length) {
        let pick = cands.reduce((a, b) => b.sil > a.sil ? b : a);
        const kept = prev && cands.find(c => c.k === prev.k);
        if (kept && kept.sil >= pick.sil - K_STICKINESS) pick = kept;
        const map = labelMap(pick.km.cents, pick.km.sizes, prev && prev.centroids);
        out.k = pick.k;
        out.silhouette = pick.sil;
        out.groups = Array.from({ length: pick.k }, (_, id) => ({ id, members: [], centroid: null }));
        pick.km.cents.forEach((c, i) => { out.groups[map[i]].centroid = c; });
        inConv.forEach((pid, i) => { const g = map[pick.km.labels[i]]; out.groupOf[pid] = g; out.groups[g].members.push(pid); });
        centroids = out.groups.map(g => g.centroid);
      }
    }

    // ---------- Per-group vote counts, consensus and repness ----------
    const G = out.groups.length;
    comments.forEach((tid, j) => {
      const c = Array.from({ length: G }, () => ({ na: 0, nd: 0, ns: 0 }));
      for (const g of out.groups) for (const pid of g.members) {
        const v = rows.get(pid).row[j];
        if (v === undefined) continue;
        c[g.id].ns++; if (v === 1) c[g.id].na++; else if (v === -1) c[g.id].nd++;
      }
      out.counts[tid] = c;
      if (G) out.consensus[tid] = {
        agree: c.reduce((p, x) => p * (x.na + 1) / (x.ns + 2), 1),
        disagree: c.reduce((p, x) => p * (x.nd + 1) / (x.ns + 2), 1),
      };
    });

    out.repness = out.groups.map(g => {
      const reps = [];
      for (const tid of comments) {
        const c = out.counts[tid], me = c[g.id];
        const rest = c.reduce((a, x, i) => i === g.id ? a : { na: a.na + x.na, nd: a.nd + x.nd, ns: a.ns + x.ns }, { na: 0, nd: 0, ns: 0 });
        for (const dir of ["agree", "disagree"]) {
          const s = dir === "agree" ? me.na : me.nd, sRest = dir === "agree" ? rest.na : rest.nd;
          const p = (s + 1) / (me.ns + 2), pRest = (sRest + 1) / (rest.ns + 2);
          const pt = propTest(s, me.ns), r = p / pRest, rt = twoPropTest(s, me.ns, sRest, rest.ns);
          if (p > 0.5 && pt > Z90 && rt > Z90) reps.push({
            tid, dir, metric: r * rt * p * pt,
            pIn: me.ns ? s / me.ns : 0, pOut: rest.ns ? sRest / rest.ns : 0, n: me.ns,
          });
        }
      }
      // Polis shows each comment once per group, in its strongest direction.
      reps.sort((a, b) => b.metric - a.metric);
      const seen = new Set();
      return reps.filter(x => !seen.has(x.tid) && seen.add(x.tid)).slice(0, MAX_REPNESS);
    });

    out.memo = {
      k: out.k,
      centroids,
      comps: Object.fromEntries(comments.map((tid, j) => [tid, [pc.comps[0][j], pc.comps[1][j]]])),
    };
    return out;
  }

  const api = { compute, propTest, twoPropTest, MIN_VOTES, MAX_K };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PolisMath = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
