# Chapter 8: LSTM vs. GRU vs. Bidirectional — Decision Guide & Common Tweaks

## The core decision tree

```
Does the task need future context (full sequence available upfront)?
├── Yes -> use a bidirectional variant (BiLSTM or BiGRU)
└── No (must process in real-time / generate left-to-right)
        │
        Do you have ample data & compute, and need maximum long-range memory?
        ├── Yes -> LSTM
        └── No (limited data/compute, or memory needs are modest) -> GRU
```

## Side-by-side comparison

| | Vanilla RNN | GRU | LSTM | BiLSTM/BiGRU |
|---|---|---|---|---|
| States carried | 1 ($h_t$) | 1 ($h_t$) | 2 ($h_t$, $c_t$) | 2x whichever base cell |
| Gates | 0 | 2 | 3 | 2x whichever base cell |
| Recurrent weight matrices | 3 | 6 | 8 | 2x whichever base cell |
| Handles long-range dependencies | Poorly (vanishing gradients) | Well | Best | Same as base cell, plus future context |
| Needs full sequence upfront | No | No | No | **Yes** |
| Real-time / streaming use | Yes | Yes | Yes | **No** |
| Training speed (relative) | Fastest | Faster than LSTM | Slower (more params) | Slowest (2 networks) |
| Typical use case | Short sequences, baselines | Resource-constrained tasks, similar performance to LSTM with less compute | Long sequences, maximum memory capacity needed | Full-sequence tasks: NER, POS tagging, masked fill-in |

## When GRU tends to match or beat LSTM

In practice, GRU and LSTM perform comparably on many tasks — the "3 gates + cell state vs. 2 gates + no cell state" difference matters less than people expect once both are properly tuned. GRU tends to have a practical edge when:

- Training data is limited (fewer parameters → less overfitting risk)
- Compute/memory budget is tight (fewer matrices → faster training and inference)
- Sequences are short-to-medium length (LSTM's extra long-range capacity matters less)

LSTM tends to win when:

- Sequences are very long and need fine-grained control over what's remembered vs. forgotten (having separate forget and input gates, instead of one coupled update gate, gives more expressive control)
- You have enough data that the extra parameters don't cause overfitting

**Interview-ready answer:** "They're usually close in practice; GRU is a reasonable default for efficiency, LSTM when you have the data/compute budget and need maximum control over long-range memory. The right move is to try both and let validation performance decide — architecture choice here is rarely the highest-leverage decision."

## Common tweaks (know these exist, even briefly)

**Peephole connections (LSTM variant).** Standard LSTM gates only look at $x_t$ and $h_{t-1}$ — they don't see the cell state $c_{t-1}$ directly when deciding what to forget/input/output. Peephole LSTMs add $c_{t-1}$ (or $c_t$ for the output gate) as an extra input to the gate equations:

$$f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)$$

Idea: let the gates directly "peek" at the memory content itself, not just the filtered hidden state, when deciding how to update. Modest, task-dependent improvements — not a default choice, but a known variant worth naming in an interview.

**Coupled forget/input gates (LSTM variant).** Instead of learning $f_t$ and $i_t$ independently, force $i_t = 1-f_t$ — whatever fraction you forget, you input the same fraction of new content. This is a direct structural step *toward* GRU (which does exactly this, built into the update gate) while keeping LSTM's separate cell state. Fewer parameters, slightly less expressive, sometimes performs identically.

**Minimal GRU / simplified gating.** Various proposed simplifications remove the reset gate entirely, using only the update gate: $h_t = (1-z_t)h_{t-1}+z_t\tanh(W_xx_t+W_hh_{t-1}+b)$ — even fewer parameters than standard GRU, at some cost to expressiveness. Useful to know these exist on a spectrum: vanilla RNN (0 gates) → minimal GRU (1 gate) → GRU (2 gates) → LSTM (3 gates + cell state) → LSTM with peepholes (3 gates + cell state + extra connections).

**Stacked (deep) bidirectional RNNs.** Exactly what it sounds like: stack multiple BiLSTM (or BiGRU) layers vertically, same as the "deep/stacked" architecture shape from the RNN curriculum's Chapter 6, but with each layer itself being bidirectional. Layer 1's concatenated $[\overrightarrow{h_t};\overleftarrow{h_t}]$ output becomes layer 2's input sequence. Common in production NER/tagging systems — more representational depth on top of bidirectional context, at the cost of more parameters and BPTT now flowing through time, direction, *and* layers simultaneously.

## Quick gotcha for interviews

**Q: Does GRU ever have MORE parameters than LSTM?**
A: No, not for the same hidden size — GRU is structurally always cheaper (6 vs. 8 recurrent weight matrices). If someone claims otherwise, check whether they've added an unusually large embedding or output layer that's dwarfing the recurrent core.

**Q: Can you use LSTM/GRU as the base cell inside a bidirectional wrapper, or is bidirectionality its own separate architecture?**
A: Bidirectionality is a wrapper, not an architecture in its own right — "BiLSTM" and "BiGRU" both just mean "run two independent copies of that cell type, one each direction, concatenate." You could even mix directions with different cell types, though that's unusual in practice.

## What's ahead

Chapter 9 is the interview cheat sheet + Q&A for the entire LSTM/GRU/Bi- curriculum, same one-page mnemonic style as the RNN curriculum's Chapter 7's cheatsheet. Chapter 10 builds everything from scratch — NumPy, then PyTorch, verified numerically.

---

**One-line summary:** use a bidirectional variant whenever the full sequence is available upfront and future context helps; between LSTM and GRU, default to GRU for efficiency and switch to LSTM when you have the data/compute and need maximum long-range control — and know that peepholes, coupled gates, and minimal GRU all sit on the same "how many gates do you actually need" spectrum.
