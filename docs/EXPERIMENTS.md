# Experiment map

Run from the repository root with `python codes/<filename>`. Phase numbers are historical identifiers, not a required execution order. The main study starts at phase 34; earlier grids are retained where they support appendix comparisons. Output paths usually start at `runs/overnight/` and are declared in each script.

| Paper question | Drivers | Notes |
|---|---|---|
| Main quality–storage grid | `run_phase34.py` | 500 examples/task, seeds 42/43/44 |
| NF4 main reference and reruns | `run_phase47_nf4.py`, `run_phase51_nf4_seeds.py` | Seeds 42 then 17/101; separate from paired-CI grid |
| Whole-model representation storage | `run_phase21.py` | Category breakdown and generated summaries; no checkpoint export |
| Uniform/smooth scaling and full GPTQ baselines | `run_phase36_weak.py`, `run_phase37_gptq_full.py` | Full-Hessian GPTQ differs from phase-34 diagonal approximation |
| Corrected compound attention/MLP sensitivity | `run_phase42_modsens_fix.py` | Use for joint-family conclusion |
| Isolated projection sensitivity | `run_phase30.py` | Historical compound rows are not authoritative; use phase 42 instead |
| Compound multi-seed checks | `run_phase74_compound_modsens_3seeds.py`, `run_phase80_dream_compound_3seeds.py` | LLaDA and Dream |
| Bits versus steps | `run_phase83_bits_steps.py` | GSM8K n=150, MMLU n=200, steps 16/32/64 |
| Extended step sweeps | `run_phase31.py`, `run_phase52_nf4_steps.py`, `run_phase60_turboldm_steps.py`, `run_phase65_turboldm_4p1_steps.py` | Task sizes and step grids vary by script |
| GSM8K calibration-seed variance | `run_phase84_seed_ci_gsm8k.py` | n=200; seeds 17/42/101; method-specific calibration settings |
| Calibration size / stage count / residual allocation | `run_phase23.py`, `run_phase24.py`, `run_phase25.py`, `run_phase26.py` | Calibration sweeps use smaller evaluation slices |
| LM head / embeddings | `run_phase27.py`, `run_phase39_embed.py`, `run_phase72_vocab_side_quant.py`, `run_phase73_vocab_combined.py`, `run_phase76_vocab_combo_seeds.py` | Auxiliary vocabulary-side sensitivity |
| Depth and mixed-bit recipes | `run_extensions.py`, `run_phase43_recipes_ci.py`, `run_phase78_dream_recipes.py` | Original extended grids plus larger recipe checks |
| Trajectory, entropy and one-step error | `run_phase44_trajectory.py`, `run_phase49_mmlu_margin.py`, `run_phase75_trajectory_n50.py`, `run_phase77_one_step_error.py`, `run_phase82_trajectory.py` | Phase 82 logs full vocabulary vectors and is disk-intensive |
| Weight fingerprint versus sensitivity | `run_phase45_arch_fingerprint.py`, `run_phase57_fingerprint_corr.py` | Run 45 plus 30/48 before CPU-only phase 57 |
| Adaptive step control | `run_phase46_adaptive_t.py`, `run_phase50_nf4_adaptive.py`, `run_phase58_dream_adaptive.py`, `run_phase68_dream_nf4_adaptive.py` | Controller diagnostics; inspect task-specific budgets |
| Runtime throughput and reconstruction | `run_phase38_chunked.py`, `run_phase54_throughput.py`, `run_phase67_dream_throughput.py` | Keep construction peaks separate from inference peaks |
| Code-generation auxiliary tasks | `run_phase53_codegen.py` | Read evaluator behavior before interpreting execution-based metrics |

## Transfer studies

| Model family | Drivers |
|---|---|
| Dream-7B | `run_phase17.py`, `run_phase48_dream_modsens.py`, `run_phase55_nf4_dream.py`, `run_phase61_dream_nf4_seeds.py`, `run_phase64_dream_seeds.py`, `run_phase70_dream_n500_fair.py`, `run_phase80_dream_compound_3seeds.py` |
| MDLM-OWT | `run_phase16.py`, `run_phase62_nf4_mdlm.py` |
| BERT / RoBERTa / DeBERTa | `run_phase33.py`, `run_phase40_deberta_v1.py`, `run_phase56_nf4_encoders.py`, `run_phase69_nf4_deberta_v1.py` |
| LLaDA-MoE | `run_phase59_nf4_moe.py`, `run_phase66_moe_methods.py`, `run_phase71_moe_bf16_n300.py`, `run_phase79_moe_modsens.py` |
| LLaDA2.0-mini | `run_phase19.py`, `run_phase32.py`, `run_phase41_16b_attnonly.py`, `run_phase81_lazy16b_more_tasks.py` |

The transfer checks have different metrics, sample counts, and engineering limits. In particular, encoder masked-token NLL does not directly compare to generation accuracy, and large-model lazy reconstruction checks do not establish general deployment efficiency.

## Core functions

`TurboLDM_experiments_helper.py` includes model and dataset registries, `QuantConfig`, `RunConfig`, `GenerateConfig`, `prepare_stage_batches_for_model`, `quantize_model_inplace`, `turboquant_vector_quantize`, `evaluate_dataset`, and representation-storage accounting. `run_overnight.py` provides shared logging and task decoding settings. Internal imports needed by the retained drivers are included.

The release excludes superseded setup pilots, manuscript build scripts, notebook outputs, results, checkpoints, and experimental fused kernels outside the camera-ready study. See REPRODUCIBILITY.md for provenance and limitations.
