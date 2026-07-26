# Phase 16, Part 2 of 5: LSTM — Derived Gate by Gate

Roadmap: 16.1 Why RNNs + vanishing gradient [done] → **16.2 LSTM (this file)** → 16.3 GRU + sequence-to-sequence → 16.4 TCN + Attention → 16.5 Transformers + N-BEATS/DeepAR/TFT.

Part 1 proved the vanilla RNN has a genuine, formula-level flaw: gradients decay geometrically through repeated multiplication, making long-range memory unlearnable. LSTM (Long Short-Term Memory) is a specific architectural fix, published in 1997, still foundational today. We derive every gate, one at a time.

---

## 1. Symbol glossary for this file

| Symbol | Plain-English meaning |
|---|---|
| $C_t$ | the **cell state** — a NEW kind of memory, separate from the hidden state, explained in section 2 |
| $h_t$ | the hidden state (same role as Part 1, but now computed differently) |
| $f_t$ | the **forget gate** output — a number between 0 and 1 per memory slot, controlling how much OLD memory to keep |
| $i_t$ | the **input gate** output — controls how much NEW information to add to memory |
| $\tilde{C}_t$ | the **candidate** new memory content — "what COULD be added," before the input gate decides how much of it actually gets added |
| $o_t$ | the **output gate** output — controls how much of the memory to actually reveal as the hidden state |
| $\sigma(\cdot)$ | the **sigmoid function** — squashes any input into the range $(0,1)$ — explained in section 2 |
| $\odot$ | element-wise multiplication — multiply two same-shaped vectors position-by-position (NOT ordinary matrix multiplication) |
| $[h_{t-1}, x_t]$ | shorthand for "stack the previous hidden state and the current input together into one longer vector" — just a notational convenience |

---

## 2. The core idea: separate a "conveyor belt" memory from the "working" hidden state

**Plain English motivation, directly targeting Part 1's diagnosed problem:** the vanilla RNN's failure came from the hidden state being REBUILT FROM SCRATCH at every single step via a $\tanh$-squashed multiplication with $W_{hh}$ — meaning OLD information had to survive being repeatedly multiplied and squashed, over and over, which is precisely what caused it to decay geometrically. **LSTM's central innovation: introduce a SEPARATE memory pathway — the cell state $C_t$ — that information can flow through with MUCH LESS transformation, specifically avoiding repeated multiplication by a weight matrix at every step.** Think of the cell state as a **conveyor belt** running through time: information can be placed ON the belt, removed FROM the belt, or just left alone to keep riding along — largely UNCHANGED — for as long as needed, rather than being forcibly reprocessed through a squashing function at every single stop.

**The new sigmoid function, needed before we can build the gates:**
$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$
**Plain English:** like $\tanh$, sigmoid squashes any real number into a bounded range — but specifically into $(0,1)$ rather than $(-1,1)$. **Why does this specific range matter here?** Because a value between 0 and 1 can be interpreted DIRECTLY as a "gate" or a "fraction/percentage" — 0 means "let NOTHING through," 1 means "let EVERYTHING through," and values in between mean "let THIS FRACTION through." **This is precisely why LSTM uses sigmoid (not tanh) for all of its GATES** — a gate is fundamentally a "how much should I let through" decision, and sigmoid's $(0,1)$ output range is EXACTLY suited to representing that as a learnable, continuous fraction (0% to 100%), applied via element-wise multiplication ($\odot$) to control information flow.

---

## 3. Building the four gates/equations, one at a time, in the order information actually flows

**Step 1 — The Forget Gate: "how much of the OLD cell state should we keep?"**
$$
f_t = \sigma(W_f\,[h_{t-1},x_t] + b_f)
$$
**Plain English:** look at yesterday's hidden state and today's new input, and decide — for EACH individual slot/dimension of the memory — a number between 0 (forget this completely) and 1 (keep this completely). **This is a genuinely new, explicit capability the vanilla RNN never had: LSTM can LEARN to actively, deliberately DECIDE to keep specific pieces of old information around indefinitely (by learning to output $f_t$ close to 1 for those slots), rather than being FORCED to let everything decay at some fixed geometric rate the way the vanilla RNN was (Part 1, section 5).**

**Step 2 — The Input Gate and Candidate Values: "what NEW information could we add, and how much of it should we actually add?"**

Two sub-pieces, computed in parallel:
$$
i_t = \sigma(W_i\,[h_{t-1},x_t]+b_i) \qquad \tilde{C}_t = \tanh(W_C\,[h_{t-1},x_t]+b_C)
$$
**Plain English for $\tilde{C}_t$ (the candidate):** "given what just happened, here's a PROPOSED new piece of information that COULD be added to memory" — computed with $\tanh$ (squashing to $(-1,1)$, since this represents actual CONTENT/VALUES being proposed, not a gate/fraction — the same role $\tanh$ played in the vanilla RNN's hidden state, section 3 of Part 1).

**Plain English for $i_t$ (the input gate):** "a separate decision, using the SAME sigmoid-gate logic as the forget gate, about HOW MUCH of that proposed candidate should actually get written into memory" — again, a number between 0 (don't add any of this) and 1 (add all of it), per memory slot.

**Why are the CONTENT (what to propose adding) and the GATE (how much to actually add) computed SEPARATELY, as two distinct pieces?** **This is a genuinely important design choice: it lets the network learn to PROPOSE something (via $\tilde C_t$) while INDEPENDENTLY deciding whether now is actually the right moment to commit it to long-term memory (via $i_t$)** — e.g., the network might constantly compute plausible "candidate" updates at every step, but only actually GATE them into permanent memory when something genuinely significant happens (analogous, in spirit, to how a Kalman filter's gain, Phase 9 section 4.2, decided HOW MUCH to trust new information versus the prior estimate — here, LSTM learns an analogous "how much to trust/commit this new information" decision, but now as a flexible, LEARNED function rather than a formula derived from known noise variances).

**Step 3 — Updating the Cell State: combining the forget and input decisions**
$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$
**This is THE single most important equation in the entire LSTM architecture, and the direct, explicit fix for Part 1's vanishing gradient problem — let's derive exactly why.**

**Plain English reading:** "the new cell state equals (how much of the OLD cell state we decided to KEEP, $f_t\odot C_{t-1}$) PLUS (how much of the NEW candidate information we decided to ADD, $i_t\odot\tilde C_t$)." **Notice the CRITICAL structural difference from the vanilla RNN's hidden-state equation (Part 1, section 3): this cell-state update is a simple, direct ADDITION of two terms, with NO forced squashing (no $\tanh$ wrapped around the WHOLE expression) and NO fixed, shared weight matrix multiplying the ENTIRE previous state at every single step the way $W_{hh}h_{t-1}$ did in the vanilla RNN.**

**Why this specific structural difference fixes the vanishing gradient problem (the actual derivation, connecting directly back to Part 1, section 5):** recall the vanishing gradient arose because backpropagating through many time steps required repeatedly MULTIPLYING by a $\tanh$-derivative term (always $<1$) and a shared weight matrix, at EVERY single step, causing exponential/geometric decay. **In the cell-state equation above, if the forget gate $f_t$ happens to be close to 1 (the network having LEARNED "this information should be preserved"), the gradient can flow BACKWARD through this ADDITION operation nearly UNCHANGED — an addition's gradient simply passes straight through both of its input terms with a multiplier of exactly 1, rather than being crushed through a repeated squashing-and-multiplying operation.** **This creates what's often called a "gradient superhighway" or "constant error carousel" — a path through time where information (and correspondingly, the LEARNING GRADIENT flowing backward through that same path) can travel across MANY time steps with dramatically LESS decay than the vanilla RNN's hidden state permitted, PRECISELY because addition (unlike repeated multiplication-and-squashing) does not inherently shrink things geometrically.** **This is the complete, precise, first-principles answer to "why does LSTM solve the vanishing gradient problem" — not a vague appeal to "it's more complex," but a specific structural fact: the cell state's update is additive and gate-controlled, rather than being forced through a repeated multiply-and-squash operation at every single step.**

**Step 4 — The Output Gate: "how much of the cell state should we actually REVEAL as the hidden state right now?"**
$$
o_t = \sigma(W_o\,[h_{t-1},x_t]+b_o) \qquad h_t = o_t \odot \tanh(C_t)
$$
**Plain English:** the cell state $C_t$ is the LSTM's full, rich internal memory — but we don't necessarily want to expose ALL of it, in raw form, as the actual working hidden state used for the current prediction. **The output gate $o_t$ (yet another sigmoid-based "how much to let through" decision) controls how much of the (tanh-squashed, to bound it back into a sensible range for onward use) cell state actually gets revealed as $h_t$** — the network can maintain rich, detailed internal memory in $C_t$ while only exposing the SPECIFIC PART that's relevant RIGHT NOW as the actual working hidden state.

---

## 4. Putting all four pieces together: the complete LSTM cell, in one place

$$
f_t = \sigma(W_f[h_{t-1},x_t]+b_f) \quad\text{(forget gate: how much old memory to keep)}
$$
$$
i_t = \sigma(W_i[h_{t-1},x_t]+b_i), \quad \tilde{C}_t=\tanh(W_C[h_{t-1},x_t]+b_C) \quad\text{(input gate + candidate: what new info to propose and how much to add)}
$$
$$
C_t = f_t\odot C_{t-1}+i_t\odot\tilde{C}_t \quad\text{(cell state update: the additive memory highway)}
$$
$$
o_t = \sigma(W_o[h_{t-1},x_t]+b_o), \quad h_t = o_t\odot\tanh(C_t) \quad\text{(output gate: how much memory to reveal right now)}
$$

**A genuinely useful, memorable summary sentence tying all four gates together, worth being able to say fluently in an interview: "At every time step, LSTM decides what to FORGET from its existing memory, what NEW information to ADD to that memory, updates the memory via simple, gradient-friendly ADDITION rather than forced multiplication-and-squashing, and finally decides how much of that memory to actually REVEAL as its current output — with each of these four decisions being a separately learned, sigmoid- or tanh-based function of the current input and previous hidden state."**

---

## 5. A small numerical taste: one LSTM step, with simplified scalar (single-number) gates

To keep this hand-computable, imagine a SIMPLIFIED LSTM with just ONE memory slot (a single number, not a whole vector) — real LSTMs have many slots simultaneously, but the mechanics per-slot are identical, so this single-slot version demonstrates the real computation faithfully.

Suppose: previous cell state $C_{t-1}=5.0$, and (having already computed the weighted sums and applied sigmoid/tanh) suppose we're given: $f_t=0.9$ (forget gate says "keep 90% of old memory"), $i_t=0.3$ (input gate says "only accept 30% of the new candidate"), $\tilde C_t = 4.0$ (the proposed new candidate content), $o_t=0.6$ (output gate says "reveal 60% of the memory").

**Cell state update:**
$$
C_t = f_t\times C_{t-1} + i_t\times\tilde C_t = (0.9)(5.0)+(0.3)(4.0) = 4.5+1.2=5.7
$$
**Interpretation: the memory mostly PERSISTED (kept 4.5 out of the original 5.0, since $f_t=0.9$ was high) with a modest new contribution added on top (1.2, since $i_t=0.3$ only accepted a fraction of the proposed 4.0) — the memory GENTLY GREW from 5.0 to 5.7, rather than being forcibly overwritten or aggressively squashed.**

**Hidden state (output):**
$$
h_t = o_t \times \tanh(C_t) = 0.6\times\tanh(5.7)
$$
$\tanh(5.7)$ is very close to 1 (since $\tanh$ saturates near $\pm1$ for large-magnitude inputs — recall from Part 1, section 3, $\tanh$'s output range is always bounded within $(-1,1)$), so approximately:
$$
h_t \approx 0.6\times 0.9998 \approx 0.5999
$$
**Interpretation: even though the internal cell-state memory has grown to a fairly large value (5.7), the EXPOSED hidden state is deliberately restrained to about 0.6 by the output gate — a concrete illustration of section 3, Step 4's point: the cell state can hold rich, larger-magnitude internal memory while only revealing a controlled, bounded portion of it as the actual working output.**

---

## 6. Quick self-check questions

1. Precisely why does using SIGMOID (rather than tanh) for the gates make sense, given what a "gate" is supposed to represent?
   *(Answer: sigmoid squashes any input into the range (0,1), which can be directly interpreted as a fraction/percentage — 0 meaning "block completely," 1 meaning "let everything through" — exactly matching what a gate is conceptually supposed to represent; tanh's (-1,1) range doesn't have this same "fraction of information to let through" interpretation.)*
2. What is the single specific structural feature of the cell-state update equation that fixes the vanishing gradient problem, and why?
   *(Answer: the cell-state update is an ADDITION of two gated terms (f_t⊙C_{t-1} + i_t⊙C̃_t) rather than a repeated multiplication-and-squashing operation; when the forget gate f_t is close to 1, gradients can flow backward through this additive path across many time steps with far less decay, since addition passes gradients through largely unchanged, unlike the vanilla RNN's forced multiply-by-W_hh-and-squash-by-tanh operation at every single step.)*
3. Why are the candidate content ($\tilde{C}_t$) and the input gate ($i_t$) computed as two SEPARATE quantities, rather than combined into one single equation?
   *(Answer: separating them lets the network independently learn WHAT information could potentially be added (the candidate, via tanh) from HOW MUCH of it should actually be committed to memory right now (the gate, via sigmoid) — allowing the network to constantly propose plausible updates while selectively deciding when those updates are actually significant enough to commit to long-term memory.)*
4. What is the role of the output gate, and why doesn't the network just expose the full cell state directly as the hidden state at every step?
   *(Answer: the output gate lets the network maintain rich, potentially large-magnitude internal memory in the cell state while only revealing the specific, currently-relevant PORTION of that memory as the actual working hidden state used for predictions at this particular time step — allowing memory storage and memory usage to be controlled somewhat independently.)*

---

## What's next
**Part 3 of Phase 16** covers the **GRU (Gated Recurrent Unit)** — a simplified, more computationally efficient cousin of LSTM that merges some of these four gates into fewer, more streamlined equations (we'll derive exactly how, and discuss the genuine practical trade-off of when to prefer GRU over LSTM) — plus **sequence-to-sequence architectures**, the standard approach for producing MULTI-STEP-AHEAD forecasts from a recurrent network (directly extending the "unroll your own forecasts forward" idea from Phase 6, Part 5, section 4.2, into the RNN/LSTM setting).

Say "next" for Part 3, or ask for more LSTM gate-derivation drilling first (e.g., a second full numerical example, or tracing through how a specific gradient would flow backward through two consecutive LSTM cells).
