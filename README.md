# Collective Inquiry demo

A static prototype of a multi-question open poll, recreated from the [ci-polis demo](https://ci-polis.vercel.app/demo). Participants answer several questions about one topic, vote on each other's responses, and see where opinion groups agree, disagree, or split.

Everything runs in the browser on synthetic data. Nothing is sent anywhere. State is kept in `localStorage`, and "Reset demo" restores the starting data.

## Live demos

- Oakland data centers (`main`): https://thebloomproject.github.io/ci-polis-demo/
- SFUSD classroom technology (`sfusd-demo`): https://thebloomproject.github.io/ci-polis-demo/sfusd/

Both are rebuilt by `.github/workflows/pages.yml` whenever either branch is pushed.

## Run it locally

```bash
python3 -m http.server 8766
```

Then open http://localhost:8766.

## Files

- `index.html`: the app (UI, seed content, synthetic participants).
- `polis-math.js`: a plain-JS port of the Polis opinion math, kept separate so it can later be swapped for results from a Polis server:
  - PCA with sparsity-aware projection;
  - k-means with k chosen by silhouette score;
  - representative statements and group-aware consensus.

## How the demo works

- **Synthetic participants** lean toward one of two hidden camps. The opinion groups in the UI are not those camps: they're discovered from the votes by `polis-math.js`.
- **Claims** are a layer on top of Polis: each groups related responses and is labelled Most agree, Most disagree, or Divisive from per-group support.
- **Votes** are Agree, Disagree, Unsure, or Need info. Unsure and Need info count as a pass in the math. Need info also feeds the "What we need to learn" list.
- **Open questions** ("What should everyone know?", "What should be done?", "What are we not talking about?") have no claims.
- **Reflection pauses** appear after 8 votes, then every 12.
- **Proposed questions** join the conversation once every opinion group rates them a good question at least 70% of the time.

Seed content and background facts are samples, not verified facts.
