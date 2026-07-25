# Chapter 3 — Packaging & Containerization

*(Module 2 of the syllabus)*

---

## 1. The gap between "a trained model" and "a running service"

At the end of Chapter 2, you have a model artifact sitting in a registry — essentially a file full of learned weights. That file, by itself, does nothing. It cannot answer an HTTP request. It cannot even be loaded without the *exact* right combination of libraries (the correct version of PyTorch/TensorFlow, the correct Python version, the correct system libraries like CUDA for GPU models).

**Packaging** is the process of turning "a file of weights" into "a runnable, callable service." **Containerization** is the specific, dominant technique used to do this reliably. Let's build up to why containers are the answer, rather than just naming the tool.

---

## 2. The "it works on my machine" problem, for models

Imagine you trained a model on your laptop. It uses library version X, a specific OS, specific system-level dependencies. Now you want that model running on a production server — possibly one with a different OS, different pre-installed libraries, maybe a different Python version entirely.

Two things can go wrong, and both are common in practice:

1. **It doesn't run at all** — missing dependency, version mismatch, crash on startup.
2. **It runs, but gives different answers** — this is the scarier one. Subtle differences in library versions (especially numerical libraries) can produce slightly different floating-point results. For most software this is irrelevant; for an ML model, a "slightly different" computation can occasionally tip a prediction across a decision boundary. This ties directly back to the reproducibility checklist from Chapter 2 — the *environment* is one of the five things you must pin, and packaging is where you actually enforce that pin.

**Containers solve this by packaging the model together with its entire runtime environment** — the exact libraries, the exact system dependencies, the exact OS-level pieces — into a single, portable unit that behaves identically no matter what machine runs it.

---

## 3. What a container actually is (plain language)

Think of a container as a lightweight, self-contained box that includes:
- Your model file
- Your serving code (the code that loads the model and exposes an API)
- Every library your code needs, at pinned versions
- A minimal slice of an operating system, just enough to run everything

The key property: this box runs *identically* whether it's on your laptop, a teammate's laptop, a staging server, or a production cluster with a thousand machines. You're no longer trusting that "the server has the right stuff installed" — you're shipping the right stuff *inside* the same unit as your code.

### Docker, specifically
Docker is the dominant tool for building and running these boxes. Two concepts matter for interviews:

- **Image** — the packaged, static blueprint (your model + code + dependencies, frozen). This is what gets versioned and stored (often right alongside the model registry entry from Chapter 2 — the registry may literally point to a specific image).
- **Container** — a *running instance* of an image. You can start many containers from the same image (e.g., 50 replicas of your model serving the same traffic in parallel).

- **Layers** — an image is built in layers (base OS layer, then dependencies layer, then your code layer). This matters practically: if only your model file changes but the base OS and dependencies stay the same, only the top layer needs rebuilding — making iteration fast. This is a common "why do we structure Dockerfiles this way" follow-up question.

### GPU-specific packaging nuance
For models that need GPU acceleration, the base image must include the correct GPU driver-compatible libraries (e.g., CUDA/cuDNN versions matched to both the GPU hardware and the ML framework version). This is a very real, very common source of production breakage — worth mentioning if a question touches on GPU serving, since it signals real deployment experience rather than textbook knowledge.

---

## 4. Serialization formats — how the model itself gets saved

Separately from containerization, there's the question of *how the model weights themselves are saved to disk*. This is a different layer from the container, but interviewers sometimes blur the two, so keep them distinct in your head:

- **Container** = the whole runtime environment (OS + libraries + code)
- **Serialization format** = just the model weights file format, which lives *inside* that container

Two considerations when picking a format:
1. **Framework lock-in** — a model saved in a framework-native format (e.g., a PyTorch-specific format) generally needs that same framework to load it.
2. **Portability via ONNX** — ONNX is an open, framework-neutral format that many frameworks can export to and many serving systems can run. The point of ONNX: train in whatever framework you like, then convert to a common format so your serving infrastructure doesn't need to support every training framework separately. This is a good answer if asked "how would you support serving models that different teams trained in different frameworks?"

---

## 5. Model servers — the piece that actually answers requests

Packaging gets your model into a runnable box. But you still need something *inside* that box whose job is: receive a request, run the model, return a prediction, and do this efficiently for many simultaneous requests. That's a **model server**.

Three you should recognize by name and by what problem each is known for solving:

| Model server | Primarily used with | What it's known for |
|---|---|---|
| **TensorFlow Serving** | TensorFlow models | Mature, production-grade serving with versioned model directories (can hot-swap model versions without downtime) |
| **TorchServe** | PyTorch models | Native PyTorch serving, similar versioning/management features for the PyTorch ecosystem |
| **NVIDIA Triton** | Multi-framework (TF, PyTorch, ONNX, etc.) | Framework-agnostic — can serve models from *different* frameworks side-by-side; strong at GPU optimization and **dynamic batching** |

**Dynamic batching**, worth understanding on its own: instead of running the model once per incoming request (inefficient, especially on GPU), the server waits a few milliseconds to collect several incoming requests, then runs them through the model *together* as one batch, because GPUs are far more efficient processing a batch than processing requests one at a time. This directly trades a tiny bit of added latency (the wait to accumulate a batch) for a large gain in throughput — a concrete, concrete example of the latency-vs-throughput tradeoff we'll formalize in Chapter 6.

---

## 6. Putting it together — the full packaging picture

```
 Model artifact (from registry, Ch2)
        │
        ▼
 Serialize to a format (framework-native or ONNX)
        │
        ▼
 Wrap with serving code (loads model, exposes predict API)
        │
        ▼
 Build container image (code + serving logic + pinned dependencies + OS slice)
        │
        ▼
 Run via a model server (TF Serving / TorchServe / Triton) inside that container
        │
        ▼
 Ready to be deployed (Chapter 5)
```

---

## 7. Common pitfall interviewers listen for

Don't conflate "container" and "model server" — a container is the portable environment; the model server is the software running *inside* it that actually serves predictions. You could theoretically run a model server without a container (just install everything directly on a machine), but you'd lose the portability/consistency guarantees — which is exactly why almost nobody does that in real production systems.

---

## Comprehension check

1. In your own words, what specific problem does containerization solve that simply "installing the same libraries on the production server" does not fully solve?
2. What's the difference between a serialization format (like ONNX) and a container — why are these two separate concerns?
3. Explain dynamic batching, and why it's a latency/throughput tradeoff rather than a pure win.

Answer if you'd like, or say "c4" to move to **Chapter 4: Training-Serving Skew**.
