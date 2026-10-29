# Reproducing the camera-ready study

This guide describes the released code, including differences between the reported paper environment and the environment available when preparing this release. It does not claim that the full GPU study has been rerun for this release.

## Environment and inputs

Use the installation commands in the README, Python 3.12, and a CUDA GPU. The paper reports an RTX 5090 (32 GB), PyTorch 2.11, CUDA 13.0, and Transformers 4.55. The release preparation environment has PyTorch 2.11.0+cu128 (CUDA 12.8); the pinned Python packages come from that environment. Kernel versions, GPU drivers, allocator behavior, and upstream model code can change results and memory measurements.

Run `python scripts/capture_environment.py > environment.local.json` and save `python -m pip freeze` alongside your locally generated logs. Set `HF_HOME`, `HF_HUB_CACHE`, and `HF_DATASETS_CACHE` explicitly to writable paths. Authenticate using `HF_TOKEN` if needed. Models using custom Hugging Face classes load with `trust_remote_code=True`.

The main model is `GSAI-ML/LLaDA-8B-Instruct`. The helper's `HF_MODELS`, `MODEL_SPECS`, and `DATASET_SPECS` registries contain the other model and dataset identifiers. Transfer paths include `Dream-org/Dream-v0-Instruct-7B`, `kuleshov-group/mdlm-owt`, `inclusionAI/LLaDA-MoE-7B-A1B-Instruct`, and `inclusionAI/LLaDA2.0-mini`. Encoder-specific IDs are in the corresponding drivers. Upstream revisions were not pinned by the original experiments; exact historical snapshots cannot be recovered from this release alone. Record resolved Hugging Face snapshot commits for your run and use those same cached snapshots for comparisons.

## Main protocol

`run_phase34.py` contains the authoritative main-grid settings:

| Setting | Released implementation |
|---|---|
| Model and precision | LLaDA-8B-Instruct, BF16 loading |
| Evaluation | First 500 rows of each task's registered split; no random evaluation subsampling |
| Tasks | MMLU test (`cais/mmlu`, `all`), ARC-Challenge test (`allenai/ai2_arc`), HellaSwag validation, GSM8K test (`main`) |
| Seeds | 42, 43, 44 |
| Methods | BF16, uniform W3/W4, diagonal GPTQ-4, AWQ-4, Q-Diffusion-Text-4, TurboLDM-(3+1)/(4+1) |
| Calibration | Up to 24 nonempty WikiText-103 validation texts after filtering short strings; separate from the four accuracy tasks |
| Calibration tokens | Main driver 192 maximum per text; paper describes a maximum of 256 |
| Calibration probes | 3 examples; four masking stages for TurboLDM |
| TurboLDM blocks and stages | 128-element blocks; stage ratios 0.85 / 0.60 / 0.35 / 0.15 |
| Residual schedule | 1-bit residual on sensitivity-ranked layer/stage pairs; top fraction 0.15 |
| Exclusions | Main-driver regex excludes embeddings, LM head, router and specified gate weights; large output heads also have helper safeguards |

Calibration takes early usable texts from a bounded source slice rather than a random sample. The helper's generic defaults differ from the main driver (for example 128 calibration examples and 512 tokens), so reproduce from the phase driver instead of constructing an unconfigured `QuantConfig()`.

The main calibration is distinct from MMLU/ARC/HellaSwag/GSM8K evaluation. Auxiliary masked-token transfer experiments use WikiText-based protocols and should be interpreted separately; do not infer the four-task separation claim for every transfer script.

`mcq_fast_gen_cfg` in `run_overnight.py` sets the task decoding budgets:

| Task type | Steps | Generated tokens | Block length | Batch size |
|---|---:|---:|---:|---:|
| Main multiple-choice tasks | 16 | 8 | 8 | 2 |
| GSM8K | 64 | 128 | 32 | 1 |

The helper defaults to zero temperature, no classifier-free guidance, and low-confidence remasking. Formatting, choice parsing, GSM8K number extraction, and per-example correctness logging are in the helper. Some transfer drivers use native decoding and override these settings; consult their `gen_cfg` records. Accuracy is stored as a fraction and displayed as a percentage. Masked-token NLL for MDLM/encoders is a separate metric, not four-task generation accuracy.

## Commands and outputs

Run the main commands in README.md from the checkout root. `run_phase34.py` produces `runs/overnight/phase34_main_ci_n500.jsonl` with 96 expected cells. Each cell contains configuration, score, storage accounting, timing, and per-example correctness. Run `scripts/check_main_grid.py` before interpreting the table. An initial phase-34 full grid is estimated at roughly 8.4 hours in its source comments; hardware, downloads and quantization overhead can change this substantially. The full appendix is a much larger workload.

NF4 is a separate experiment: phase 47 runs seed 42, and phase 51 adds seeds 17 and 101. The paper reports identical scores under these deterministic reruns. They are not included in the paired-CI comparison; do not merge them into phase 34 as if they shared its seed grid.

Phase 21 writes per-category CSV accounting and LaTeX summaries into generated `papers/` paths. Its historical `papers/neurips2026` mirror is retained for script compatibility and is created automatically. These paths are local generated outputs, not shipped manuscript content.

Most drivers append JSONL and many skip cells already present. **Run only one driver at a time on one GPU.** Resume keys and behavior vary by phase: read each driver's skip logic. For a clean comparison, archive your existing `runs/overnight/` outside the checkout and start with an empty output directory. Changing a configuration without changing the output path can otherwise cause an old cell to be reused. Never run the same phase concurrently against the same JSONL.

Drivers often catch exceptions and append to `failures.jsonl`; a zero process exit code does not guarantee a complete grid. Inspect failures and per-cell coverage. The release's main-grid validator checks method/task/seed/model, duplicate cells, sample counts, correctness arrays, and score consistency. Historical failure entries can remain after successful recovery.

## Analysis and statistics

```bash
mkdir -p papers
python scripts/check_main_grid.py
python codes/bootstrap_ci.py \
  --in runs/overnight/phase34_main_ci_n500.jsonl \
  --out papers/gen_table_main_ci_n500.tex
```

The bootstrap script pairs correctness arrays with the BF16 row for the same task and seed, pools the resulting differences across seeds, and runs 2,000 bootstrap resamples with a NumPy generator seeded at zero. It reports across-seed mean and sample standard deviation and a 95% interval for the gap in percentage points. This is the released pooled-example procedure, not a hierarchical seed bootstrap. Its pairing assumes unchanged dataset order and formatting; the historical arrays do not carry independently verified example IDs. Missing arrays produce unavailable CIs, so do not accept a table without validating its input grid.

For auxiliary tables and plots, first generate the inputs mapped in EXPERIMENTS.md, then run:

```bash
mkdir -p papers
python codes/aggregate_and_render.py
python codes/render_new_tables.py
python codes/render_pareto_steps.py
python codes/render_trajectory_fig.py
python codes/render_phase82_84.py
```

These are historical analysis scripts covering different subsets of the study. They do not rebuild the complete camera-ready manuscript, which includes editorial synthesis and inline tables. Missing inputs may be skipped or produce partial outputs. Check their input filenames and generated captions before using a figure. No manuscript compiler is included because the release has no manuscript sources.

Phase 82 records full-vocabulary logits and can generate multi-gigabyte JSONL; run it only when the detailed trajectory analysis is needed. Phases 44/75 are narrower trajectory diagnostics. Phase 57 needs the phase-45 fingerprint CSV and sensitivity logs from phases 30/48.

## Storage versus live memory

`model_parameter_storage_bytes` and `quant_storage_bytes` account for compressed codes and metadata, together with unquantized weights where appropriate. Sub-byte codes are accounted at their logical packed widths. These numbers are not a measurement of a `.pt` file produced by `torch.save`, and the experiment drivers do not export reusable packed checkpoints. Python/tensor representation overhead and dequantized runtime tensors must not be inferred from nominal storage bits.

The main reference quantizers reconstruct floating-point weights, so compact representation accounting does not imply equivalent live VRAM savings. NF4 uses bitsandbytes `Linear4bit` with BF16 compute. Phase 38 contrasts chunked reconstruction, phase 54 measures end-to-end throughput/memory, and the large-model lazy path is a diagnostic with additional limitations. Quantization/loading peaks can exceed the reported evaluation peak, which is reset around evaluation. Keep serialized representation, resident tensors, peak inference memory, and construction peak distinct.

## Provenance and release validation

`source-manifest.json` records original working-tree and released file hashes. The original working tree includes small CPU-quantization memory fixes to the helper; its Git HEAD alone does not identify those bytes. Release edits replace author-specific cache paths, create missing generated directories, and correct the reusable bootstrap caption. The original workspace is left untouched.

Release checks cover Python syntax, internal imports, offline CPU codec behavior, generated GIF readability, and synthetic validator/bootstrap inputs. They do not establish full numerical reproduction, throughput equivalence, or that every upstream model remains loadable. The paper remains the reference for its reported observations; rerun the corresponding GPU drivers to evaluate them on your hardware.
