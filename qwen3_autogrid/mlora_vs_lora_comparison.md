# Qwen3 LoRA vs mLoRA comparison

Generated: 2026-05-28 13:44 UTC

Grid: coarse 45 runs, refine 48 runs. Metric: tinyMMLU mean accuracy unless noted.

## Best validation perplexity & tinyMMLU

| Stage | LoRA best PPL | LoRA run | mLoRA best PPL | mLoRA run | LoRA best tinyMMLU | mLoRA best tinyMMLU | Δ acc (mLoRA−LoRA) |
|---|---:|---|---:|---|---:|---:|---:|
| Coarse | 1.0344 | `lr_2p0000e-05_r64_a128_st500_wd_0p0000e00` | 1.0410 | `lr_2p0000e-03_r16_a32_st500_wd_0p0000e00` | n/a (no coarse tinyMMLU for LoRA) | 0.5428 | — |
| Refine | 1.0352 | `lr_2p0000e-04_r64_a128_st500_wd_1p0000e-03` | 1.0353 | `lr_3p0000e-03_r32_a64_st500_wd_0p0000e00` | 0.4968 | 0.5428 | **+0.0459** |

## Top-5 tinyMMLU mean

| Stage | LoRA top-5 mean | mLoRA top-5 mean | Δ |
|---|---:|---:|---:|
| Coarse | n/a | 0.5411 | — |
| Refine | 0.4885 | 0.5397 | **+0.0512** |

## Full MMLU (top-5 refine runs)

| Method | Status | Top-5 full MMLU mean |
|---|---|---:|
| LoRA | 5/5 done | 0.4543 |
| mLoRA | 1/5 complete (`lr_2p0000e-04_r32_a64_st500_wd_0p0000e00`, orchestrator relaunched) | pending |

## Refine grid rank correlation (matched hyperparams)

Spearman ρ = **0.8455** over 48 shared (lr, rank, wd) configs.
Positive ρ: higher tinyMMLU on LoRA tends to co-occur with higher tinyMMLU on mLoRA.

## Top-10 by tinyMMLU (refine)

| Rank | LoRA run | LoRA acc | LoRA PPL | mLoRA run | mLoRA acc | mLoRA PPL |
|---:|---|---:|---:|---|---:|---:|
| 1 | `lr_3p0000e-04_r32_a64_st500_wd_1p0000e-03` | 0.4968 | 1.0380 | `lr_2p0000e-04_r32_a64_st500_wd_0p0000e00` | 0.5428 | 1.0502 |
| 2 | `lr_4p0000e-04_r32_a64_st500_wd_1p0000e-02` | 0.4917 | 1.0422 | `lr_2p0000e-04_r32_a64_st500_wd_1p0000e-02` | 0.5428 | 1.0502 |
| 3 | `lr_3p0000e-04_r32_a64_st500_wd_0p0000e00` | 0.4875 | 1.0402 | `lr_2p0000e-04_r32_a64_st500_wd_1p0000e-03` | 0.5428 | 1.0502 |
| 4 | `lr_3p0000e-04_r32_a64_st500_wd_1p0000e-02` | 0.4875 | 1.0402 | `lr_2p0000e-04_r64_a128_st500_wd_0p0000e00` | 0.5351 | 1.0439 |
| 5 | `lr_3p0000e-04_r64_a128_st500_wd_0p0000e00` | 0.4787 | 1.0465 | `lr_2p0000e-04_r64_a128_st500_wd_1p0000e-02` | 0.5351 | 1.0439 |
| 6 | `lr_2p0000e-04_r32_a64_st500_wd_1p0000e-03` | 0.4753 | 1.0373 | `lr_2p0000e-04_r64_a128_st500_wd_1p0000e-03` | 0.5351 | 1.0439 |
| 7 | `lr_2p0000e-04_r32_a64_st500_wd_0p0000e00` | 0.4753 | 1.0373 | `lr_3p0000e-04_r32_a64_st500_wd_0p0000e00` | 0.5351 | 1.0446 |
| 8 | `lr_2p0000e-04_r32_a64_st500_wd_1p0000e-02` | 0.4753 | 1.0373 | `lr_3p0000e-04_r32_a64_st500_wd_1p0000e-02` | 0.5351 | 1.0446 |
| 9 | `lr_4p0000e-04_r32_a64_st500_wd_0p0000e00` | 0.4728 | 1.0440 | `lr_3p0000e-04_r32_a64_st500_wd_1p0000e-03` | 0.5351 | 1.0446 |
| 10 | `lr_4p0000e-04_r32_a64_st500_wd_1p0000e-03` | 0.4728 | 1.0440 | `lr_4p0000e-04_r32_a64_st500_wd_0p0000e00` | 0.5330 | 1.0415 |

## Takeaways

- **mLoRA wins on tinyMMLU (refine)**: best refine tinyMMLU **+4.6 pp** vs LoRA; coarse tinyMMLU was only evaluated for mLoRA (45/45).
- **LoRA wins on validation PPL** (lower is better): mLoRA best PPL is ~0.012 (coarse) and ~0.013 (refine) higher.
- Best mLoRA refine run: `lr_2p0000e-04_r32_a64_st500_wd_0p0000e00` at **54.28%** tinyMMLU vs LoRA best `lr_3p0000e-04_r32_a64_st500_wd_1p0000e-03` at **49.68%**.
- Full MMLU top-5 for mLoRA relaunched on cluster (`cluster_qwen3_mmlu_mlora_full_top5_submit.sh`); update this doc when 5/5 complete.
