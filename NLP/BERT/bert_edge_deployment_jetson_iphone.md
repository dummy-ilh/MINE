# Training & Serving BERT on Small Devices — Jetson Nano & iPhone, End to End

Tenth companion doc. Open-ended, full pipeline: train where you have compute, compress and convert for the target device, serve with a real tokenizer and inference runtime, and know the actual numbers you're budgeting against. Structured as one shared front half (training/compression, device-agnostic) and two device-specific back halves (Jetson Nano, iPhone), since that's genuinely how the work splits.

---

## Part 0 — The core principle that shapes everything below

**You never train on the device. You train (or fine-tune) on a real GPU, then compress and convert for the device.** Jetson Nano and iPhones are inference targets, not training targets — Nano's 4GB shared RAM and lack of a real training-grade GPU, and iPhone's thermal/battery/OS constraints, make on-device training impractical for anything beyond tiny personalization tweaks (e.g. Core ML's on-device personalization APIs for last-layer updates, which is a narrow, different use case from what's covered here). The pipeline is always: **cloud/desktop training → compression → device-specific conversion → on-device serving.**

---

## Part 1 — Model Selection (device-agnostic)

**Don't start from BERT-base for either device.** Start from something already compressed and validated for edge deployment:

| Model | Params | Relative size | Notes |
|---|---|---|---|
| BERT-base | ~110M | baseline | Reference point only — not a realistic edge deployment target |
| DistilBERT | ~66M | ~60% of BERT-base | Good general starting point (full build details in the DistilBERT doc) |
| MobileBERT | ~25M | ~23% of BERT-base | Purpose-built for mobile — bottleneck structures, inverted-residual-style blocks inside each layer, designed explicitly for latency on phone-class hardware |
| TinyBERT | ~14.5M (4-layer variant) | ~13% of BERT-base | Aggressive distillation (including intermediate-layer distillation, not just output), best for very tight latency budgets at some further accuracy cost |

**Practical recommendation:** start with **DistilBERT** for prototyping (best accuracy/effort trade-off, most tooling support), then move to **MobileBERT or TinyBERT** if profiling on the actual target device shows you need more headroom — don't pre-optimize before you've measured on real hardware.

---

## Part 2 — Train / Fine-Tune (device-agnostic, do this on a real GPU)

Follow the fine-tuning practices from the earlier companion docs (hyperparameters doc, layer-freezing doc) unchanged — this step doesn't know or care yet that the model is headed to a small device. One addition specific to eventual edge deployment:

**Consider quantization-aware training (QAT) instead of post-training quantization (PTQ), if accuracy after PTQ isn't good enough.**

- **PTQ** (simpler, try this first): fine-tune normally in fp32, then quantize the finished model afterward (see Part 3). No changes to the training loop.
- **QAT** (more effort, better accuracy at low precision): simulate quantization's rounding effects *during* fine-tuning, so the model learns weights that are robust to the precision loss it will actually experience at inference time. Reach for this specifically if PTQ's accuracy drop is unacceptable for your task — a common escalation path, not a default starting point.

---

## Part 3 — Compression (device-agnostic, applies before either device-specific conversion)

**Quantization** — reduce numeric precision of weights (and often activations):

| Precision | Size vs fp32 | Typical accuracy impact | Notes |
|---|---|---|---|
| fp32 (baseline) | 100% | none | Training default, too large/slow for either target device |
| fp16 | 50% | negligible | Good default for Jetson Nano (has real fp16 hardware support) |
| int8 | 25% | small, usually 1-3% depending on task and calibration quality | Best for both devices when latency/memory is tight; needs either PTQ calibration data or QAT |

**Pruning** (optional, layer-dependent effort/payoff): removing individual attention heads or weights found to contribute little. Worth doing if you've already distilled/quantized and still need more headroom — diminishing and less predictable returns compared to distillation + quantization, so treat it as a later-stage lever, not a first move.

---

## Part 4 — Jetson Nano: Conversion & Serving

### 4a. Nano's actual constraints (budget against these numbers)

- 4GB RAM **shared** between CPU and GPU (not dedicated VRAM) — this is the single tightest constraint, tighter than compute in most cases.
- Maxwell-generation GPU, ~472 GFLOPS fp16 — meaningfully weaker than a modern discrete GPU; expect fp16/int8 to matter a lot here, not just as a nice-to-have.
- No dedicated ML accelerator (unlike iPhone's Neural Engine) — everything runs through the GPU or CPU.

### 4b. Conversion pipeline: PyTorch → ONNX → TensorRT

**Step 1 — Export to ONNX** (verified pattern, tested end-to-end against a small BERT-style model):

```python
import torch

model.eval()  # always eval() before export -- BatchNorm/Dropout behave differently otherwise

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch"},
    },
    opset_version=17,
)
```
Confirmed working end-to-end in testing: exported cleanly, passed ONNX's structural checker, and produced correct output shapes when run through `onnxruntime`.

**Step 2 — Build a TensorRT engine from the ONNX file** (on the Nano itself, or cross-compiled — TensorRT engines are hardware-specific, so build for the actual Nano, not a dev desktop GPU):

```bash
trtexec --onnx=model.onnx \
        --saveEngine=model.trt \
        --fp16 \
        --minShapes=input_ids:1x1,attention_mask:1x1 \
        --optShapes=input_ids:1x64,attention_mask:1x64 \
        --maxShapes=input_ids:1x128,attention_mask:1x128
```

**Why `--fp16`:** the single highest-value flag on Nano's hardware — real fp16 tensor throughput, roughly half the memory of fp32, and typically negligible accuracy loss. For int8, you additionally need a calibration step (`--int8` plus a representative calibration dataset) since int8 requires knowing realistic activation ranges, not just a data-type cast.

**Why explicit min/opt/max shapes:** TensorRT engines are shape-specialized for performance — giving it a realistic range (tied to the max_length decision from the hyperparameters doc, not BERT's default 512) avoids either over-allocating for sequences you'll never see or failing at runtime on a length outside the built range.

### 4c. Serving on Nano

- **Runtime:** TensorRT's C++ or Python inference API, or `onnxruntime` with the TensorRT execution provider if you want to skip manual engine-building complexity at some performance cost.
- **Tokenizer:** run on CPU, not GPU — tokenization is lightweight relative to the model forward pass, and keeping it off the GPU avoids contending with the model for Nano's limited GPU resources. Hugging Face's `tokenizers` library (Rust-backed) is fast enough here without needing GPU acceleration.
- **Batch size:** typically 1 for real-time/interactive use cases (voice command, live classification) — Nano's memory and compute budget rarely justifies batching multiple requests together the way a server would.
- **Realistic expectation to budget against:** a well-quantized DistilBERT/MobileBERT-scale model at seq_len ~64-128 typically lands in the tens-of-milliseconds range per inference on Nano with fp16 — treat this as a ballpark to validate against your own profiling, not a guarantee, since it depends heavily on your specific model size, sequence length, and how well-tuned the TensorRT engine build is.

### 4d. Nano-specific pitfalls

- **Thermal throttling** — Nano has no active cooling in its default form factor; sustained inference load can throttle clock speeds after continuous use. If deploying for sustained workloads, budget for a fan/heatsink and validate latency under sustained load, not just a cold-start benchmark.
- **Shared memory pressure** — the OS, any other running processes, and the model all compete for the same 4GB. Leave real headroom; don't size your model/batch to Nano's theoretical maximum.
- **First-inference latency** — TensorRT engines can have a slower first call (kernel selection/warm-up); if latency-sensitive, run a dummy warm-up inference at startup before serving real requests.

---

## Part 5 — iPhone: Conversion & Serving

### 5a. iPhone's actual constraints (budget against these)

- **Apple Neural Engine (ANE)** — a dedicated ML accelerator, not just CPU/GPU — this is the biggest structural difference from Nano. Getting inference to actually run *on* the ANE (rather than falling back to CPU/GPU) is the main performance lever, and it depends on using operations Core ML can map to ANE.
- **App bundle size limits** — matters for App Store distribution; a full BERT-base at ~440MB (fp32) is a real consideration for app size, another reason to compress before shipping, independent of latency.
- **Battery/thermal budget** — background or frequent inference needs to be power-conscious in a way a plugged-in edge device doesn't.

### 5b. Conversion pipeline: PyTorch → ONNX (or directly via TorchScript) → Core ML

**Step 1 — Export to ONNX or TorchScript** (ONNX export pattern is identical to the Nano path above — same verified code). `coremltools` can convert from either ONNX or a traced TorchScript model; TorchScript tracing is often the more reliable path in practice for Transformer-family models with dynamic control flow.

**Step 2 — Convert to Core ML with `coremltools`:**

```python
import coremltools as ct

mlmodel = ct.convert(
    traced_model,  # torch.jit.trace(model, (dummy_input_ids, dummy_attention_mask))
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, 128))),
        ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(1, 128))),
    ],
    compute_units=ct.ComputeUnit.ALL,       # let Core ML choose CPU/GPU/ANE per-op automatically
    minimum_deployment_target=ct.target.iOS16,
)
mlmodel.save("BertClassifier.mlpackage")
```

**Why `compute_units=ALL` and not forcing ANE directly:** Core ML doesn't let you force specific ops onto the ANE — you give it the model and deployment target, and its compiler decides per-operation which unit (CPU/GPU/ANE) to run on, falling back automatically for ops the ANE doesn't support. Your job is making choices that *make ANE placement likely* (see next point), not commanding it directly.

**Step 3 — Quantize/palettize for size and ANE-friendliness:**

```python
import coremltools.optimize.coreml as cto

op_config = cto.OpPalettizerConfig(nbits=8)  # or use linear quantization instead of palettization
config = cto.OptimizationConfig(global_config=op_config)
compressed_model = cto.palettize_weights(mlmodel, config)
```

**Why palettization specifically (Core ML's version of weight clustering), not just linear int8 quantization:** Core ML supports both; palettization (clustering weights into a small lookup table of values) is particularly well supported for reducing on-disk model size for App Store distribution and tends to be well-optimized by Apple's runtime — linear quantization is also viable and simpler to reason about, so both are worth benchmarking on your actual model rather than assuming one is strictly better.

### 5c. Serving on iPhone

- **Tokenizer:** Core ML doesn't include a tokenizer — you need a Swift-native or bundled tokenizer implementation (e.g. a Swift port of WordPiece tokenization, or Apple's `swift-transformers` package which includes tokenizer support alongside Core ML model loading). This is a genuinely separate integration step people forget when they think "convert the model" is the whole job.
- **Inference call:** through Core ML's generated Swift model class (Xcode auto-generates a typed Swift interface from the `.mlpackage`) — straightforward `let output = try model.prediction(input_ids: ..., attention_mask: ...)` once the model and tokenizer are both integrated.
- **Batch size:** 1, same reasoning as Nano — interactive, single-request, on-device use cases rarely benefit from batching.
- **Realistic expectation to budget against:** a well-converted, ANE-eligible DistilBERT/MobileBERT-scale model at short sequence lengths typically runs in the low tens-of-milliseconds range on recent iPhone hardware when actually placed on the ANE — again a ballpark for profiling comparison, not a promise, and it depends heavily on which iPhone generation you're targeting as your minimum supported device.

### 5d. iPhone-specific pitfalls

- **Silent CPU/GPU fallback** — if part of your model uses an operation the ANE doesn't support, Core ML silently falls back to CPU/GPU for that portion rather than failing loudly. Always profile with Xcode's Core ML performance report (shows per-op compute unit placement) rather than assuming ANE placement happened just because you set `compute_units=ALL`.
- **Dynamic shape ranges cost you.** `RangeDim` flexibility (supporting a range of sequence lengths in one model) is convenient but can prevent some ANE optimizations that a fixed shape enables — if your app's use case has a genuinely fixed or narrow sequence-length need, consider a fixed-shape model instead for better ANE eligibility.
- **Minimum deployment target trade-off** — the newer the `minimum_deployment_target` you require, the more Core ML optimizations become available, at the cost of excluding users on older iOS versions — a genuine product decision, not just a technical one.
- **On-device model updates** — if you'll want to update the model post-launch without a full app release, look into Core ML model deployment via CloudKit-hosted model updates, a separate mechanism from bundling the model directly in the app.

---

## Part 6 — Shared end-to-end checklist

1. Pick a starting model (DistilBERT to start; MobileBERT/TinyBERT if profiling later demands it).
2. Fine-tune on a real GPU using standard practices (hyperparameters/layer-freezing docs) — PTQ by default, QAT only if PTQ's accuracy drop is unacceptable.
3. Decide your real max sequence length from your actual data's distribution (not a device-driven guess) — this affects every downstream conversion step's shape configuration.
4. Quantize (fp16 baseline, int8 if you need more headroom and can calibrate/QAT for it).
5. Export to ONNX (same code, verified, for both device paths) or TorchScript (iPhone-preferred path).
6. **Nano:** ONNX → TensorRT engine (`trtexec` with `--fp16`/`--int8`, explicit shape ranges) → serve via TensorRT or `onnxruntime` + TensorRT execution provider.
7. **iPhone:** ONNX/TorchScript → Core ML via `coremltools` (`compute_units=ALL`) → palettize/quantize for size → serve via the Xcode-generated Swift model class + a real tokenizer integration.
8. Profile on the *actual target device*, not a dev machine — Nano's thermal behavior and iPhone's ANE placement are both things you cannot reliably predict from desktop benchmarks.
9. Validate accuracy *after* every compression step (quantization, palettization/pruning) against a held-out set — compression is not free, and the point of profiling on-device is partly to catch an unacceptable accuracy/latency trade-off before shipping, not after.
