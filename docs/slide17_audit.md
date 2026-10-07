# Slide 17 Architecture Audit

**Date:** 2026-10-07  
**Branch:** Improve  
**Auditor:** Kiro  

---

## What Slide 17 Specifies

```
measured Rd (f=0, f=0.1)
    │
    ▼
  MLP  (inverse: Rd → μa, μs′)         ← the new piece, replaces LUT lookup
    │
    ▼
  μa, μs′
    │
    ▼
  differentiable Cuccia or PCBC         ← forward model, sits inside training loop
  (μa, μs′ → predicted Rd)
    │
    ▼
  compare with measured Rd → loss → backprop through Cuccia/PCBC into MLP weights
```

The LUT never enters training. It is used **only** for grading (every % error
on the slides comes from comparing the MLP's μa/μs′ against the LUT answer).

**Key property — self-supervised:** no labelled (μa, μs′) targets are needed.
The only supervision signal is the measured Rd itself, fed back through the
differentiable physics formula. This is why the method can work where nobody
has built a look-up table for a specific setup.

---

## What Is Currently Implemented

| Component | File | Status |
|---|---|---|
| Inverse MLP (Rd → μa, μs′) | `sfdi/pinn.py` — class `PINN` | ✅ Complete |
| Cuccia forward model (differentiable) | `sfdi/physics.py` — `rd_cuccia` | ✅ Complete |
| PCBC forward model (differentiable) | `sfdi/physics.py` — `rd_pcbc` | ✅ Complete |
| Self-supervised training loop | `sfdi/pinn.py` — `train()` | ✅ Complete |
| Huber-log loss (slide 17) | `sfdi/pinn.py` — `measurement_loss()` | ✅ Complete |
| Log-MSE loss (ablation panel) | same function, `kind="log_mse"` | ✅ Complete |
| LUT grading only (never in training) | `sfdi/experiment.py` — `grade()` | ✅ Correct |
| Training script | `scripts/03_train_pinn.py` | ✅ Complete |
| Ablation sweeps (slides 26–38) | `scripts/04_ablation.py` | ✅ Complete |
| Forward MLP (μa, μs′ → Rd) | `sfdi/forward_mlp.py` | ✅ Added this PR |
| Forward MLP training script | `scripts/00b_train_forward_mlp.py` | ✅ Added this PR |

**The Slide 17 pipeline is fully implemented.**  
`sfdi/pinn.py`'s `PINN` class is the inverse MLP.

---

## Architecture Verification: `pinn.py` vs Slide 17

### Network shape
- Input: 2 (Rd at f=0, Rd at f=0.1)  
- Hidden: 3 × 128 with SiLU — matches the ablation panel ("widths 128,128,128, SiLU")
- Output: 2 (log μa, log μs′ before exponentiation)

### Input encoding
```python
z = self.net(torch.log(R.clamp_min(1e-6)))   # log input
```
Log-transforms the measured Rd before feeding the network.  
Output is squashed through a sigmoid into the configured (μa, μs′) bounds, so the
network can never predict outside the physics range.

### Loss (slide 17, Eq. shown on slide)
```python
r = log(Rd_predicted) − log(Rd_measured)
huber_log:  0.5 * Huber_{δ=0.03}(r).sum(-1).mean()
```
This matches the slide's Huber loss over log-Rd ratios, δ = 0.03.

### Training hyperparameters (from config.yaml — match ablation panel)
| Setting | Value |
|---|---|
| Widths | [128, 128, 128] |
| Activation | SiLU |
| Optimiser | AdamW, lr 0.003 |
| LR decay | × 0.9997 per step (ExponentialLR) |
| Batch size | 32 768 |
| Steps | 10 000 |
| Seed | 42 |
| Sampling | uniform over all selected pixels |

All match slide 17 and the ablation panel.

---

## What the Forward MLP (`forward_mlp.py`) Is

This is a **separate** neural network that learns the *forward* mapping:
```
(μa, μs′) → Rd
```
It is trained to reproduce the LUT's values (supervised, using LUT points as
labels). It is **not** the Slide 17 inverse MLP, and it does **not** appear
in the current PINN training loop.

Potential uses (none currently wired up):
1. As a differentiable replacement for Cuccia/PCBC inside the training loop —
   could learn the LUT's implicit model rather than an analytic formula.
2. As a fast vectorised forward pass for other downstream tasks.

---

## Known Gaps / Things to Ask the PL

| # | Gap | Impact |
|---|---|---|
| 1 | **Refractive index A** — config says `A_source: fresnel, n: 1.4`. The comment reads "ASK PL: tissue refractive index in their code". Wrong A shifts every μs′ estimate. | Medium |
| 2 | **Boundary parameter B** — PCBC in the slide may use a "calibrated boundary parameter B" (not the Fresnel A). `A_value` in config is null. | Medium for PCBC runs |
| 3 | **ForwardMLP not wired into PINN** — `forward_mlp.py` exists but the PINN still uses the analytic Cuccia/PCBC. If the goal is to use the MLP as the differentiable forward model, that integration is not done. | Low (current results use analytic formulas, which is fine) |
| 4 | **No trained ForwardMLP weights** — `results/real/forward_mlp.pt` does not exist yet; `get_forward_mlp()` will raise `FileNotFoundError` if called. Run `scripts/00b_train_forward_mlp.py` to produce it. | Low until gap 3 is addressed |
| 5 | **Why does the lab want the MLP at all?** — Current results show the MLP at best ties the LUT's grader on these slides. The self-supervised advantage (no labelled targets, works without a pre-built table) is not demonstrated on a setup that lacks a LUT. Confirming the motivation would clarify whether gap 3 needs to be closed. | Low — ask PL |

---

## Summary

The Slide 17 inverse-MLP pipeline is **fully implemented and working**. The
PINN trains self-supervised (no μa/μs′ labels), uses differentiable Cuccia or
PCBC as the physics check, and is graded exclusively by the LUT. The two files
added in this PR (`forward_mlp.py`, `00b_train_forward_mlp.py`) provide a
stand-alone forward-direction network that could later replace the analytic
formulas in the training loop, but that wiring is not needed for the current
Slide 17 results.
