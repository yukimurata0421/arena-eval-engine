# MCMC Parallelism and GPU Policy

Status: accepted
Decision date: 2026-06-03 JST
Scope: Dell real-data workstation, Stage 4/5 probabilistic workloads, public `arena run`
Source of truth: `src/arena/lib/platform_setup.py`, `src/arena/pipeline/stages.py`, `src/arena/pipeline/backend.py`

## Purpose

This document fixes how ARENA handles pipeline workers, MCMC chains, CPU/GPU selection, and current hardware notes.

## Current Hardware

Current Dell workstation:

```text
CPU: Intel Xeon E5-2695 v4 @ 2.10GHz
CPU threads: 36
cores/socket: 18
threads/core: 2
GPU: NVIDIA GeForce GTX 1070
GPU memory: 8192 MiB
driver: 580.159.03
```

Older benchmark notes referred to a Core i7-class workstation with GTX 1060. The current real-data workstation is the Xeon E5-2695 v4 / GTX 1070 machine above, so old wall-clock expectations should not be reused without re-measurement.

## JAX / GPU State

The current machine has NVIDIA hardware, but JAX does not expose a CUDA backend in the current Python runtime.

Observed state:

```text
GPU hardware: NVIDIA GeForce GTX 1070
JAX backend: cpu
JAX devices: CPU only
CUDA backend exposed to JAX: no
```

Pipeline log wording:

```text
GPU: Not available to JAX -> CPU
NVIDIA hardware found, but JAX did not expose a CUDA device
```

## Decision

Default behavior:

```text
pipeline workers:
  use all logical CPU threads unless --workers is set
  current Dell default: 36

MCMC chains:
  default: 4

MCMC host device cap:
  default: 12

GPU:
  not forced for small-N models
  use CPU when n <= 5000 unless force flags are set
```

Environment knobs:

| env | meaning |
| --- | --- |
| `ADSB_PHASE_CHAINS` | Stage 4 phase evaluator chains |
| `ADSB_BAYES_PHASE_CHAINS` | Bayesian phase comparison chains |
| `ADSB_CP_CHAINS` | single change point chains |
| `ADSB_MCP_CHAINS` | multiple change point chains |
| `ADSB_MCMC_MAX_WORKERS` | MCMC host worker/device cap |
| `ADSB_GPU_MIN_N` / `ARENA_GPU_MIN_N` | minimum N before GPU is considered |
| `ADSB_FORCE_GPU` / `ARENA_FORCE_GPU` | force GPU trial |
| `ADSB_DISABLE_GPU` / `ARENA_DISABLE_GPU` | force CPU |

## Pipeline Execution Method

Current `arena run` processing:

```text
Stage 1:
  wave-parallel aggregation

Stage 2 / 4 / 5 / 7 / 8:
  cross-stage parallel group when workers > 1

Stage 3:
  remains outside the cross-stage group because it consumes Stage 2 outputs

Stage 9:
  runs after analysis/proxy outputs and builds evidence synthesis artifacts
```

This means `--workers 36` controls outer pipeline concurrency, not MCMC chain count.

## Why Workers and Chains Are Separate

These are different controls:

| control | purpose | default |
| --- | --- | --- |
| pipeline workers | concurrent script execution | `os.cpu_count()` -> 36 on current Dell |
| MCMC chains | posterior sampling chains | 4 |
| JAX host devices | CPU device parallelism for JAX/NumPyro | capped at 12 |

Letting pipeline workers flow directly into MCMC chains made small-N probabilistic stages slower without improving decision quality enough to justify the cost.

## Real-Data Measurement

Measured on the current Dell real-data workstation:

| workload | before | after |
| --- | ---: | ---: |
| Full `arena run` | `232s (3.9 min)` | `171s (2.9 min)` |
| Bayesian phase comparison | `102.0s` | `51.7s` |
| Single change point | `148.5s` | `67.8s` |
| Multiple change point | `184.5s` | `94.0s` |
| Phase Bayes comparison | `166.7s` | `126.9s` |

The full run improved by about 26%.

Public-repo verification on 2026-06-03 completed the full Stage 1-9 real-data
pipeline with 37 OK in 174 s (2.9 min). That run used the same Xeon E5-2695 v4
host, GTX 1070 hardware, JAX CPU fallback, `--workers 36`, and default 4-chain
MCMC policy.

## Accuracy Check

The main accuracy concern is Bayes MC precision after reducing chains.

Stage 4 Phase Bayes comparison:

```text
chains=4 vs chains=12
max effect mean difference: 0.61 percentage point
max P(>0) difference: 1.0 point
supported / unclear judgment: no change
Traffic elasticity: 0.0053 both
Minutes elasticity: 0.5097 vs 0.5100
```

This does not prove `chains=4` is always enough. It means the current real-data comparison did not change ARENA's decision.

## Follow-up

Next convergence hardening:

```text
1. run default chains=4
2. check R-hat / ESS / divergence
3. rerun with chains=8 or chains=12 only when convergence is poor
4. write diagnostics into EvidenceRow.diagnostics and warnings
```

## Anti-Patterns

Avoid:

```text
- using GPU just because physical NVIDIA hardware exists
- equating --workers with MCMC chains
- treating current chains=4 validation as a universal proof
- hiding convergence issues behind faster wall-clock time
```
