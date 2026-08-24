# Moshi Grounding — Experiment Ledger

**North star:** final model stays a **full-duplex interactive** Moshi; grounding must make the live CONVERSATION better AND preserve base quality. Offline metrics necessary, not sufficient.
**Reference (base moshiko):** live fluency **speech-fraction = 0.406**; recognition majority-baseline varies by family.

**Two-line story so far:**
1. **Grounding-as-tokens** (state written into the inner-monologue text stream) lifts offline recognition (emotion bal-acc up to **0.74**) but **breaks the live conversation** — the model over-emits tokens and goes silent (speech-fraction 0.00–0.28). Offline-recognition and live-fluency are in *tension*.
2. **Grounding-as-conditioning** (emotion/arousal/dominance as LUT conditioner inputs, sum-fused into the transformer, never in the speech stream) **preserves clean speech** (real words, no token-spam) and is the right approach. Best config recovers **~60% of base fluency** (100% DailyTalk + emotion); a gap to base remains (see headline).

## ⭐ HEADLINE (robust fluency n=24, 2026-08-20) — best conditioning ≈ 60% of base; still a gap
**⚠️ Metric-variance lesson:** speech-fraction at n=6 is unreliable (S_cond_v3 read 0.370 then 0.155 then 0.232 across runs — per-prompt is bimodal: a prompt either engages or stays silent). Use **n≥24**. Numbers below are n=24 (base re-measured same run).

| run | config | speech-frac (base = **0.390**) | reading |
|---|---|---|---|
| **S_cond_v3** | 100% DailyTalk, emotion | **0.232** | ★ best — talks on most prompts, silent on ~40% |
| **S_cond_cog** | DailyTalk, emo+arousal+dominance (SER) | **0.207** | cognitive stack ≈ emotion-only (no fluency cost) |
| S_cond_iempad | IEMOCAP, pad-weight .1 | 0.161 | best IEMOCAP variant |
| S_cond_v2 | 75% DailyTalk | 0.126 | more Fisher → lower |
| S_cond_iemL | IEMOCAP, 1600 | 0.086 | |
| S_cond_mix | DailyTalk+IEMOCAP | 0.078 | **mix did not help** |
| S_cond_iem | IEMOCAP, 800 | 0.073 | |

**Honest conclusions:**
1. **Conditioning preserves clean, emotion-aware speech** (real words, no token-spam) — the approach is right. Best recipe = **100% DailyTalk + emotion conditioning** (S_cond_v3), ~**0.23 vs base 0.39 (≈60%)**. Conversational, but still quieter than base on a subset of prompts. Not "solved."
2. **Cognitive conditioning is free**: emo+arousal+dominance (S_cond_cog 0.207) ≈ emotion-only (v3 0.232). We can add cognitive state without hurting fluency. ✅
3. **NEGATIVE RESULT — IEMOCAP did not help fluency** (iem/mix/iemL all 0.07–0.09, below DailyTalk-only). Hypothesis "long dyadic turns fix brevity" **not supported** — likely the reconstructed *acted* dyadic audio (overlaps/resample/scripted) is lower-quality training signal than DailyTalk. `iempad` (low pad-weight) is the only IEMOCAP variant that helps, and only modestly.
4. **Data-mix direction holds but weakly**: v3(100%DT) 0.232 > v2(75%) 0.126. More DailyTalk helps; but pure-DailyTalk still leaves a gap to base.
5. **Levers** (padlo/hilr/r128, n=6): none beat plain v3; low pad-weight helps IEMOCAP only.

**✅ QUALITY GATE — S_cond_v3 PASSES (audio_CE, 2026-08-20):** in-domain (DailyTalk) audio_CE **1.677 vs base 2.069 (−19%, better)**, text_CE 1.248 vs 1.795, cb0 0.962 vs 1.439 — no voice-quality loss, model predicts in-domain speech better while carrying emotion conditioning. OOD (Libri) mild specialization: audio_CE 5.300 vs 5.082 (+4.3%), cb0 3.235 vs 3.186 (+1.5%), text_CE 4.208 vs 3.777 (+11%) — small, not the material rise the gate watches for. **Verdict: "still a good model" ✓.** (Gate fix: `quality_eval.py` now builds+passes `condition_tensors`; a conditioning model's `lm.forward` asserts fuser/conditions present → blank-message crash without them.) Remaining check: live **affect-congruence** A/B.

**⚠️ AFFECT-CONGRUENCE A/B — WEAK/NEGATIVE (S_cond_v3, 2026-08-20):** feed the same 14 user prompts under COND_EMOTION∈{neu,hap,ang,sad}, score response affect with audeering SER. Result: affect barely moves — arousal flat 0.711–0.714, valence sad .508/neu .529/hap .528/**ang .469** (anger→lower valence + higher dominance is the ONLY real directional signal; magnitude small). Diagnostic confirms it's NOT a plumbing bug: the emotion LUT learned **distinct** per-emotion condition vectors (L2 ≈ 2.0 apart, norms ~1.4), and LMGen bakes `fuser.get_sum(condition_tensors)` at each `reset_streaming` so the swap reaches generation. **The model is emotion-aware at the INPUT but not the OUTPUT — it isn't yet reliably emotion-steerable at cfg_coef=1.** (Eval fix: swap `lm_gen.condition_tensors` per emotion + reset_streaming; do NOT rebuild InferenceState on a shared mimi → "already streaming" assert.)
**→ CFG SWEEP (cfg∈{1,3,6}, 2026-08-20) — PARTIAL POSITIVE:** guidance amplifies the emotion signal on the **valence** axis. Response valence (SER), hap−ang separation: cfg1 **+0.068** → cfg3 +0.020 → cfg6 **+0.158** (>2× cfg1). At **cfg=6**: hap valence 0.552 (highest) vs ang 0.394 (lowest); hap is top-valence at every cfg. **⇒ the learned emotion condition IS behaviorally usable, just needs amplification (cfg≈6).** Caveats: only valence responds (arousal/dominance don't separate cleanly; ang even reads lower arousal at cfg6 — wrong dir); effect modest+noisy, non-monotonic (cfg3 dips). Tool: `affect_ab.py --cfg`.
**→ QUALITY-UNDER-CFG (2026-08-20) — UNFAVORABLE TRADEOFF:** cfg=6 **halves fluency**: speech-fraction (n=24) base 0.407, v3@cfg1 0.216, **v3@cfg6 0.103 (−52%)**. So strong guidance buys the valence gain by cutting speech ~half. Since cfg3 *worsened* congruence and cfg6 *hurts* fluency, **CFG alone cannot deliver both emotion-steering AND fluency** — the guidance that steers is the guidance that silences. `fluency_audio.py --cfg` added.
**→ To get STRONG steering (beyond CFG):** stronger conditioner training — higher conditioner LR / condition upweight / cleaner (less noisy than SER) emotion labels / more steps; or a stronger injection point than sum-fusion. IEMOCAP's human VAD labels could help label quality here despite hurting fluency — worth a targeted conditioner-only run.

**⇒ CONDITIONER-UPWEIGHT SWEEP (COND_LR_MULT param-group, 2026-08-21):** conditioner params train at `lr×MULT`, rest at base lr. **up10 (10×) — PARTIAL WIN:** fluency **0.182** (n=24; vs v3 0.216 — mild cost, no collapse) AND behavioral steering **emerges at cfg=1** (no fluency-killing CFG). Affect A/B cfg=1: arousal hap **0.751** > sad 0.706 (✅, vs v3's FLAT 0.711–0.714), dominance ang>sad ✅ — BUT valence scrambled (ang 0.527 highest, hap 0.424 — wrong dir ✗). So upweight moves steering to cfg=1 cheaply, but on arousal/dominance not valence. up30/up100 pending (~02:44/02:52) test whether more upweight fixes valence or costs more fluency. Train loss up10 1.111 (< v3). Mechanism: `train.py` `COND_LR_MULT` env; eval auto-runs via `moshi_exp/upweight_results.txt`.
**up30 (30×) — fluency best-yet 0.252 (n=24, > v3 0.216), but steering WEAKER:** valence still wrong (ang 0.579 highest, hap 0.515 lowest ✗), and arousal hap−sad margin collapsed 0.045→0.006 (noise). So MORE upweight did NOT help steering (arguably hurt) while fluency stayed fine. **Emerging conclusion: upweight magnitude is not the lever — affect differences are small/inconsistent and valence is reliably wrong at 10× and 30×.** Likely cause is deeper: noisy SER emotion labels can't teach a clean happy→positive mapping, and/or sum-fusion is too weak an injection. up100 + upcog30 pending; if valence still doesn't move, next lever = **label quality** (e.g. condition on the continuous SER valence axis, or IEMOCAP human valence) not more upweight.
**up100 (100× emo-only):** fluency 0.218 (≈v3), valence STILL fails (neu highest). **upcog30 (30× emotion+arousal+dominance) — ★ FIRST VALENCE PASS:** hap 0.535 > neu 0.493 > sad 0.460 ✅ at cfg=1; fluency 0.148 (lowest of sweep, steering/fluency tradeoff; still talks). **KEY INSIGHT: it's not upweight MAGNITUDE, it's DEDICATED DIMENSIONAL conditioners.** Emotion-only at 10/30/100× never fixed valence; adding explicit arousal+dominance channels disentangles the axes and valence ordering falls out. Categorical 4-class emotion muddles valence; dimensional conditioning separates it. Caveats: modest spread (~0.075), small n (5-9/cond), anger still anomalous on arousal. **⇒ validates the dimensional-conditioning direction → S_cond_val (explicit valence conditioner) in pipeline should push it cleaner.**

**★ S_cond_val (emotion + VALENCE conditioner, 30× upweight) — DIRECT VALENCE STEERING PASS (2026-08-21):** the label-quality run — condition on the continuous SER valence axis (lo/mid/hi buckets) instead of categorical emotion. Result at cfg=1: **fluency 0.254** (best-tier, no cost) AND **valence steering monotonic PASS** — COND_VALENCE lo→mid→hi gives output valence 0.446 → 0.484 → 0.490 (hi>mid>lo). Spread modest (0.044; really lo-clearly-lower, mid/hi saturate), small n (8-11/bucket), but the FIRST direct, monotonic, fluency-free valence control. **CONCLUSION: the steering problem was LABEL QUALITY, not architecture/upweight — categorical 4-class emotion never steered valence; the dimensional/continuous label does.** Recipe forward = v3 conditioning architecture + DIMENSIONAL conditioners (valence/arousal/dominance) + modest upweight. Next to strengthen magnitude: more upweight on valence / cleaner labels (human VAD) / CFG on the valence axis. Bug fixed en route: `run_inference.py` default map lacked `valence` → fluency eval crashed silently on emotion+valence models (now `"valence":"mid"`). Data `DailyTalkCondVal` (valence dist lo~130/mid~275/hi~230), builder `build_conditioned_val.py`, eval `valence_ab.py`.
**S_cond_val follow-ups (val_probe, 2026-08-21):** (1) **CFG does NOT amplify dimensional valence — it DEGRADES it:** spread cfg1 +0.044 → cfg3 +0.013 → cfg6 **−0.073 (inverts)**. Opposite of categorical-emotion valence (which cfg6 amplified). So for dimensional conditioners the amplification lever is upweight/labels, NOT CFG. (2) **S_cond_val PASSES quality gate:** audio_CE ID **1.69 < base 2.07** (better), OOD 5.30 vs 5.08 (+4%, mild) — same clean profile as v3. **⇒ S_cond_val is the new best deployable candidate: fluent 0.254 + quality-preserving + valence-steerable (v3 + working valence control).**
**★ S_cond_val100 (100× upweight) — UPWEIGHT AMPLIFIES:** valence spread **0.079** (vs 30×'s 0.044 — nearly 2×) and now **cleanly monotonic** (lo 0.474 < mid 0.517 < hi 0.553, ~+0.04/step); fluency held **0.223**. **⇒ LEVERS SETTLED: upweight AMPLIFIES dimensional valence, CFG DEGRADES it.** Recipe = dimensional conditioner + high conditioner-LR upweight. Not saturating at 100× → `S_cond_val300` (300×) queued to find the ceiling / fluency cost.
**★ S_cond_full (emotion+arousal+dominance+valence, 30×) — BEST VALENCE STEERING:** valence spread **0.118** (monotonic 0.348→0.416→0.466), vs valence-only val 0.044 (30×) / val100 0.079 (100×). Fluency 0.150 (4-conditioner stack + upweight costs speech). **CONFIRMS: more DIMENSIONAL conditioners → wider valence steering (disentanglement), even at lower upweight.** Categorical emotion in same model still fails valence. **val300 (300×) — BROKE:** loss 2.695 (no convergence), babble output, valence inverted → **upweight ceiling ≈100×** (100× coherent/0.079; 300× destabilizes). Config-sed gotcha: `s|S_cond_val|S_cond_iemval|` on `S_cond_val100` yields `S_cond_iemval100` — watch substring seds.

## Valence-steering scoreboard (spread hi−lo, cfg=1, base fluency 0.406)
| model | conditioners | upweight | spread | fluency |
|---|---|---|---|---|
| S_cond_val | emotion+valence | 30× | 0.044 | 0.254 |
| S_cond_val100 | emotion+valence | 100× | 0.079 | 0.223 |
| **S_cond_full** | emo+aro+dom+valence | 30× | **0.118** | 0.150 |
| S_cond_val300 | emotion+valence | 300× | broke (−0.02) | babble |
| S_cond_iemval | emo+valence, HUMAN labels | 100× | INCONCLUSIVE | 0.116 |
Steering/fluency tradeoff is real: wider spread costs speech. Sweet spot TBD (full stack at lower upweight? partial stack?).
**S_cond_iemval (human valence) — INCONCLUSIVE, audio confound:** fluency 0.116 (IEMOCAP acted audio) → only ~8/42 responses scorable → **n=2-4/valence bucket** = noise (measured -0.143, meaningless). Human labels NOT tested — confounded by IEMOCAP's poor training audio (same fluency hit as all IEMOCAP runs). **The human-label route via IEMOCAP is a dead end** (no human valence labels exist on clean DailyTalk audio). Label-quality ceiling question stays open but unanswerable this way.

**⚠️ MULTI-AXIS PROBE on S_cond_full (axis_ab.py, 2026-08-22) — WEAK/NOISY + variance warning:** arousal +0.028 (DIR-OK), dominance −0.021 (FAIL), valence **−0.080 (FAIL)** — but the SAME checkpoint measured valence **+0.118** 2h earlier. Not real: S_cond_full's low fluency (0.150) leaves only **4–8/14 prompts scorable** → tiny-n noise, number swings sign. **LESSONS: (1) the full-stack 0.118 was NOT robust (small-n luck); (2) low-fluency models can't be reliably eval'd for steering (too few scorable responses) AND aren't usable — so a FLUENT model is the critical path; (3) steering eval needs n≥24 on a fluent model.** Reliable results are the n=24 ones on fluent models only: S_cond_val 0.044, val100 0.079 (monotonic). Multi-axis (arousal/dominance) independent control NOT yet demonstrated — re-test on a fluent sweet-spot model (full10/full50) at n≥24.

**★ SWEET-SPOT FRONTIER (full stack @ 10/30/50×, n=24, 2026-08-22) — FULL STACK REJECTED:** fluency full10 0.183 / full30 0.150 / full50 0.172; valence spread full10 **0.016 (flat)**, full50 **FAIL (inverted)**, full30's 0.118 was n=14 noise (−0.080 at n=24). **The 4-conditioner stack does NOT reliably steer valence at any upweight, and is LESS fluent than focused models.** "More dimensional conditioners → better steering" = REJECTED at n=24 (arousal+dominance muddied both fluency & valence). **WINNER = focused emotion+valence, ~100× upweight (S_cond_val100): fluency 0.223, valence spread 0.079 monotonic (n=24), quality-gate pass. Simpler beats fuller.** Note: reliable steering numbers require BOTH n≥24 AND a fluent model (≥~0.2) so enough responses are scorable — low-fluency full-stack evals were pure noise. To test arousal/dominance steering cleanly, use a FLUENT model that has those conditioners (S_cond_cog, fluency 0.207) — not the full stack.

**★ MULTI-AXIS on S_cond_cog (fluent 0.207, n=24, 2026-08-22) — ONLY VALENCE STEERS:** arousal lo 0.739→mid 0.726→hi 0.698 = FAIL/**inverted** (spread −0.041); dominance 0.743→0.717→0.728 = FAIL/flat (−0.015). Reliable (7-12/bucket, monotonic-ish, not noise). **Multi-dimensional grounding is PARTIAL: valence steers cleanly (0.079 monotonic), arousal/dominance do NOT** (arousal runs backwards, dominance flat) — even though the conditioners learn distinct vectors for all three. Likely valence is the most renderable axis in short conversational turns (positivity via word choice), while arousal/dominance (energy/assertiveness) aren't expressed in short responses and/or have weaker SER labels. **Steering capability = valence control specifically.** Future: per-axis label quality, longer responses, or a stronger injection for aro/dom.

**★ LIVE FULL-DUPLEX A/B on S_cond_val100 (live_ab.py, 2026-08-22) — REAL-TIME ✅, steering NOT robust live:** streamed 16 user turns under COND_VALENCE lo vs hi. **RTF mean 0.32-0.34 (median 0.318, max 0.71) — real-time confirmed WITH the conditioner** (~3× faster than realtime; grounding doesn't break full-duplex). Turn-taking alive (56-69% of turns produce speech). **BUT valence steering INVERTED live: lo 0.576 > hi 0.478 (n=9-11)** — opposite of offline n=24 (hi>lo +0.079). The 0.079 effect is WITHIN run-to-run noise at n~10 (stochastic gen + small scorable set). **Verdict: model is real-time + conversational, but valence steering (0.079) is too SMALL to manifest reliably in a fresh live run — likely imperceptible to a listener at this magnitude.** Gold-standard human listening test still needed; but this says the priority is WIDENING the spread, not just measuring it. Wavs saved audio_samples/live_ab/.

## OVERNIGHT SUMMARY (2026-08-22) — dimensional conditioning is the settled recipe
- **Steering mechanism SOLVED in principle:** dimensional (continuous) conditioners steer valence; categorical emotion never did. More dimensional channels → wider steering (full stack 0.118 > valence-only). Upweight amplifies (30×→100×); **ceiling ~100×** (300× breaks).
- **Open tension:** steering/fluency tradeoff — wider valence spread costs speech (full-stack 0.118 @ fluency 0.150 vs val 0.044 @ 0.254). Need the sweet spot: full stack at 50-100× vs partial stack, and whether a fluency-recovery pass (more plain DailyTalk) claws speech back.
- **Best DEPLOYABLE today:** S_cond_val (fluent 0.254 + quality-gate pass + modest valence steer 0.044) OR S_cond_v3 (emotion-aware, no explicit steer). Best STEERING: S_cond_full (0.118) but fluency 0.150.
- **Dead ends:** CFG (degrades dimensional valence), IEMOCAP (audio hurts fluency; can't test human labels), upweight >100× (breaks).

**Next to close the fluency gap:** more DailyTalk epochs on the v3 recipe; try decoding temperature / longer gen window (may be a generation-length artifact, not training); real-data long dyadic corpus better than acted IEMOCAP; then the **quality gate (audio_CE)** + **affect-congruence** live A/B on S_cond_v3 / S_cond_cog.

---

## Phase 1–2 — Token grounding (recognition up, fluency down)

| Run | What | Data | Recognition (bal-acc) | Fluency (spF) | State | Takeaway |
|---|---|---|---|---|---|---|
| emo_smoke / cond_smoke | pipeline smoke | — | — | — | ✅ done | infra validated (trains, ckpts, conditioner materializes) |
| emo_lora_01 / state16_01 | early emotion-token LoRA | DailyTalk | ~0.55 | — | ✅ done | first grounding signal; raw-acc inflated by imbalance → switched to **balanced acc** |
| grounding_stageA | emotion tokens | DailyTalk+Fisher | up | — | ✅ done | grounding learns |
| grounding_stageA_recog / **S3b** | + recognition upweight | grounded | **0.74** | **0.00** (live) | ✅ done | ⚠️ **caught by live test**: recognition 0.74 but 4/16 turns audible — upweight makes fluency WORSE |
| S4_full / S4b_full_recog | full grounding stack (emo+manner+env…) | grounded | high | low | ✅ done | dense tags crowd out speech |
| S7_up10 | 10× upweight | grounded | — | — | ❌ cancelled | pivoted to fluency-first |
| S8_fluent → **S8_final** | fluency-first, plain-heavy, sparse tags, NO upweight | plain DT+Fisher+sparse emo | — | 0.045 → **0.279** | ✅ done | talks more, but still emits some tokens aloud (spam) |
| S9_fluent_cog / S9_cog | dense cognitive tokens (aro/dom/pit) | grounded | — | **0.000** | ✅ done | ❌ fatal — dense cognitive tokens kill speech |
| S10_heavyplain | token grounding, plain-heavy mix | plain-heavy | — | 0.085 | ✅ done | token approach caps low on fluency |
| S_cog | cognitive tokens learnable (aro/dom/pit) | DailyTalk | learns aro/dom/pit | — | ✅ done @400 | cognitive dims *are* learnable as tokens |

**Verdict Phase 1–2:** token grounding is a dead-end for the live model — every variant trades speech for tokens. → switch to conditioning.

---

## Phase 3 — Conditioning approach (clean speech, brevity to fix)

Emotion/arousal/dominance as **LUT conditioner inputs** (sum-fused), never spoken. Data = plain words + `text_conditions{emotion,arousal,dominance}`. Gated by `COND_EMOTION=1` (+`COND_COG=1` for cognitive). 8 moshi+FSDP integration fixes; conditions embedded INSIDE `lm.forward` (FSDP-safe).

| Run | Data | Config | Fluency (spF) | State | Note |
|---|---|---|---|---|---|
| **S_cond32** | 33% DailyTalk + Fisher | emo only, 800 | ~0.02 (noisy) | ✅ done | Fisher-heavy → low |
| **S_cond_v2** | 75% DailyTalk | emo, 800 | **0.126** (n=24) | ✅ done | more Fisher → lower |
| **S_cond_v3** | 100% DailyTalk | emo, 800 | **0.232** (n=24) | ✅ done | **★ best recipe — ~60% of base, clean emotion-aware speech** |
| **S_cond_cog** | DailyTalk (SER emo+aro+dom) | 3-cond, 800 | **0.207** (n=24) | ✅ done | cognitive stack ≈ emo-only, no fluency cost ✅ |
| **S_cond_iem** | 🆕 IEMOCAP real-label dyadic (median 67 turns) | 3-cond, 800 | **0.073** (n=24) | ✅ done | human VAD + real timing, but **acted audio hurts fluency** |

Observed on conditioning ckpts: `Scond_neu 0.121`, `Scond_late 0.136`, `Scond2_dailytalk 0.008`. All **talk with real words** (e.g. "…and we're just so happy…") — no token-spam — but **stop early**.

---

## Phase 4 — Overnight sweep (attack brevity; distinct levers)

All 16-node, cognitive conditioning (emo+aro+dom). Fluency probes auto-fire on each checkpoint.

| Run | Lever tested | Data | Change | speech-frac | Verdict |
|---|---|---|---|---|---|
| **S_cond_mix** | combine best data | DailyTalk+IEMOCAP | — | 0.078 (n24) | ❌ mix ↓ vs DT-only |
| **S_cond_mixL** | + longer + IEM-weighted | DailyTalk+IEMOCAP×1.5 | 1600 | ~0 (noisy) | ❌ IEM-heavy hurts |
| **S_cond_iemL** | more training on IEMOCAP | IEMOCAP | 1600 | 0.086 (n24) | ❌ still low |
| **S_cond_padlo** | keep talking | mix | pad-weight .5→**.1** | **0.027** (n24) | ❌ no help on mix |
| **S_cond_iempad** | pad-lever on IEMOCAP | IEMOCAP | pad-weight **.1** | 0.161 (n24) | ✅ best IEMOCAP variant (still < v3) |
| **S_cond_hilr** | faster grounding | mix | lr **1e-5** | **0.048** (n24) | ❌ no help |
| **S_cond_r128** | more capacity | mix | rank **128** | **0.024** (n24) | ❌ no help |

**Verdict Phase 4:** hypotheses (a) long-IEMOCAP fixes brevity — **rejected** (acted dyadic audio hurts); (b) lower pad-weight — helps IEMOCAP only, modestly; (c) capacity/LR — no help. **Winner remains plain S_cond_v3 (100% DailyTalk + emotion).**

---

## Data assets
- **DailyTalkCondCog** — DailyTalk, SER (audeering) emo+aro+dom, 2541 clips (long multi-turn).
- **DailyIEMOCAPCog** 🆕 — IEMOCAP reconstructed dyadic stereo, **human** VAD labels, 1143 clips, median-67-turn dialogues, real timing. Converter `build_iemocap_dyadic.py` (uses `sphn`, not torchaudio). Best real-label conversational data we have.
- FisherConv/FisherCond — 33k short 2-turn (emotion only).
- Rejected: swb_multiturn (no user-channel audio), MELD (multi-party TV, laugh-track).

## Open items / next
- Collect fluency curve across all Phase-3/4 checkpoints (auto).
- **Quality gate**: run `quality_eval.py` (audio_CE vs base) on finished conditioning ckpts — confirm "still a good model".
- **Affect-congruence** live A/B (base vs conditioned response affect given emotional input) — the real grounding payoff, `full_duplex_test.py`.
- Pick winner by: fluency ≈ base AND audio_CE ≈ base AND congruence↑.
