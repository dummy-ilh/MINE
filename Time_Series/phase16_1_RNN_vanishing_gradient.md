# Phase 16, Part 1 of 5: Why RNNs, and the Vanishing Gradient Problem

This is the largest phase in the syllabus, so we split it into five parts:
**16.1 Why RNNs + vanilla RNN + vanishing gradient (this file) → 16.2 LSTM, fully derived → 16.3 GRU + sequence-to-sequence → 16.4 TCN + Attention → 16.5 Transformers for time series + N-BEATS/DeepAR/TFT.**

Everything here assumes zero prior deep learning background — we build up from "what is a neural network doing at all" through to the specific problem that motivated LSTM (Part 2).

---

## 1. Symbol glossary for this file

| Symbol | Plain-English meaning |
|---|---|
| $x_t$ | the INPUT to the network at time step $t$ (in a time series context, this might be $y_{t-1}$ or a feature vector) |
| $h_t$ | the **hidden state** at time $t$ — the network's internal "memory"/summary of everything relevant it has seen so far |
| $\hat{y}_t$ | the network's output/prediction at time $t$ |
| $W_{xh}, W_{hh}, W_{hy}$ | **weight matrices** — grids of tunable numbers the network learns, each one controlling how one thing influences another (subscripts tell you: FROM which quantity, TO which quantity) |
| $b_h, b_y$ | bias terms — simple constant offsets, one per equation |
| $\tanh(\cdot)$ | the hyperbolic tangent function — an "activation function" that squashes any input into the range $(-1,1)$, explained in section 3 |
| $\mathcal{L}$ (script L) | the **loss function** — a formula measuring how wrong the network's predictions are (directly generalizing the sum-of-squared-errors idea from Phase 6, Part 4) |
| $\frac{\partial \mathcal{L}}{\partial W}$ | a **gradient** — literally "how much would the loss change if I nudged this specific weight $W$ a tiny bit" — the quantity neural networks use to learn |
| $\odot$ | element-wise multiplication of two vectors (multiply matching positions together, a simple mechanical operation, used starting in Part 2) |

---

## 2. Why do we need a NEW tool at all? What's wrong with everything from Phases 1-15 for this purpose?

**Plain English motivation, connecting directly to Phase 14's core reframing:** Phase 14 showed how to turn a time series into a FIXED-SIZE table of features (a fixed number of lag columns, rolling stats, etc.) for an ordinary ML model. **This works well, but has a real limitation: you have to DECIDE IN ADVANCE exactly how many lags/how much history matters (e.g., "I'll use the last 7 lags"), and the model has NO way to automatically learn to look further back if the TRUE relevant memory length turns out to be, say, 50 steps, or to VARY depending on context.** Similarly, ARIMA (Phase 6) requires you to fix $p$ and $q$ in advance (even if chosen via AIC/BIC, Phase 6 Part 4, they're still FIXED, finite numbers once chosen).

**The core idea behind RNNs (Recurrent Neural Networks), in plain English before any formula:** instead of deciding a fixed lag window in advance, build a mechanism that processes the sequence ONE STEP AT A TIME, and at each step, maintains a running SUMMARY ("hidden state") of everything relevant it has seen SO FAR — updating that summary as new information arrives, and, crucially, LETTING THE NETWORK ITSELF LEARN (from data) how to build and maintain that summary, rather than us hand-specifying a fixed window size. **"Recurrent" simply means the network reuses the SAME computation, repeatedly, at every time step** — directly analogous, conceptually, to how every one of our recursive formulas so far (AR(1)'s $x_t=\phi x_{t-1}+\varepsilon_t$, SES's $\hat x_{t+1}=\alpha x_t+(1-\alpha)\hat x_t$, the Kalman filter's predict-update loop) reused the SAME formula at every step — **an RNN is, at its philosophical core, the same "apply the same recursive rule repeatedly" idea you've now seen many times throughout this course, except now the "rule" itself is a flexible, LEARNED function (a small neural network) instead of a fixed, simple formula with just one or two parameters like $\phi$ or $\alpha$.**

---

## 3. Building the vanilla RNN, one equation at a time

**The hidden state update equation (the heart of the RNN):**
$$
h_t = \tanh(W_{xh}\, x_t + W_{hh}\, h_{t-1} + b_h)
$$

**Breaking this down piece by piece, very slowly:**
- $x_t$: today's input (e.g., the actual observed value $y_{t-1}$ if we're using the network to forecast $y_t$, or a richer feature vector).
- $h_{t-1}$: YESTERDAY's hidden state — the running summary/memory carried forward from the previous step.
- $W_{xh}\, x_t$: a learned, weighted transformation of TODAY'S input — plain English, "how should today's raw input be reshaped/emphasized before folding it into memory." ($W_{xh}$ reads as "weight matrix, FROM $x$ TO $h$" — i.e., it governs how input $x$ influences the hidden state $h$.)
- $W_{hh}\, h_{t-1}$: a learned, weighted transformation of YESTERDAY'S memory — "how should the PAST summary be reshaped before combining it with new information." ($W_{hh}$ reads as "FROM $h$ TO $h$" — hidden state influencing the next hidden state — this is the genuinely RECURRENT piece, the network referencing its own previous output, exactly like AR(1) referencing $x_{t-1}$.)
- $b_h$: a simple additive constant (a bias term — plain English, allows the whole expression to be shifted up or down by a learned amount, giving the model one more degree of flexibility).
- $\tanh(\cdot)$: **the activation function** — defined and explained fully in the next paragraph.

**What is $\tanh$, and why do we need it at all (a genuinely important question, not just a technical detail)?** Plain English: $\tanh(z) = \frac{e^z - e^{-z}}{e^z+e^{-z}}$ is a specific S-shaped curve that takes ANY real number input and "squashes" it into the range between $-1$ and $+1$ (large positive inputs map close to $+1$; large negative inputs map close to $-1$; an input of exactly 0 maps to exactly 0). **The crucial reason we need this squashing step at all: WITHOUT it, if you just added up $W_{xh}x_t + W_{hh}h_{t-1}$ directly with no non-linear transformation, the ENTIRE recurrent network, no matter how many steps or how many layers, would mathematically collapse into being EQUIVALENT to a single, plain LINEAR model — exactly the same limitation a plain AR(p) model already has (Phase 6, Part 1's formula is inherently linear in the past values).** **The whole POINT of using a neural network instead of just AR(p) is to capture GENUINELY NON-LINEAR relationships in the data (e.g., "the effect of a large positive shock is different from the effect of an equally large negative shock" — directly recalling the leverage-effect asymmetry from Phase 10, section 2, stylized fact 3, which plain linear GARCH couldn't fully capture either) — and $\tanh$ (or similar non-linear "activation functions") is EXACTLY what injects this crucial non-linearity into the model, at every single time step.**

**The output equation (turning the hidden state into an actual prediction):**
$$
\hat{y}_t = W_{hy}\, h_t + b_y
$$
**Plain English:** "take the current hidden state (our running summary of everything relevant so far) and pass it through one more learned transformation to produce the actual forecasted number." **Notice this final step is genuinely just an ordinary LINEAR regression (Phase 7's exact machinery!) applied to the hidden state $h_t$** — all the genuinely complex, non-linear "memory-building" work happens inside the recurrent hidden-state equation above; this final output step is comparatively simple.

**How the whole thing runs, start to finish (the complete recipe, tying it together):** initialize $h_0$ (often just a vector of zeros, representing "no information yet," directly analogous to Phase 6, Part 6's initialization discussion for Holt's method). Then, for $t=1,2,3,\ldots$: compute $h_t$ from $x_t$ and $h_{t-1}$ (the hidden-state equation), then compute $\hat y_t$ from $h_t$ (the output equation), then move to $t+1$ and repeat, feeding the JUST-COMPUTED $h_t$ in as the "previous" hidden state for the next step. **This is EXACTLY the same recursive "use yesterday's output as today's input" pattern from Phase 6, Part 5, section 4's multi-step-ahead forecasting, and Phase 9's Kalman filter predict-update loop — a genuinely recurring structural theme across this entire course, now appearing again in a neural-network setting.**

---

## 4. How does the network actually LEARN the weight matrices? Building intuition for backpropagation

**Plain English, the core learning idea, directly generalizing MLE from Phase 6, Part 4:** just like MLE searched for the parameter values ($\phi,\theta$) that made the observed data most probable/minimized squared errors, a neural network searches for the WEIGHT MATRICES ($W_{xh}, W_{hh}, W_{hy}$) that MINIMIZE a loss function $\mathcal{L}$ (typically something like the sum of squared errors between $\hat y_t$ and the true $y_t$ across the whole sequence — directly the same MSE idea from Phase 13, section 5!). **The method used to actually FIND these minimizing weights is called gradient descent: repeatedly nudge each weight a small amount in whichever direction REDUCES the loss, using the GRADIENT (the "slope"/rate of change of the loss with respect to that specific weight) to know which direction to nudge.**

**New term: backpropagation.** Plain English: this is simply the specific, mechanical PROCEDURE for computing all these gradients efficiently, by working BACKWARD through the network's computations (from the final loss, back through the output equation, back through each time step's hidden-state equation, all the way back to the very first time step) — using the calculus chain rule (multiplying together a sequence of "how much does A affect B, and how much does B affect C" derivatives, to figure out "how much does A ultimately affect C" — a standard calculus tool, applied here mechanically across a long chain of time steps). **For a RECURRENT network specifically, this backward pass has to travel all the way back through EVERY single time step (since $h_t$ depends on $h_{t-1}$, which depends on $h_{t-2}$, and so on, all the way back to $h_0$) — this specific variant is called "Backpropagation Through Time" (BPTT).** You don't need to hand-derive the full chain-rule computation for interview purposes — the crucial, genuinely important consequence of this LONG backward chain is what we derive next.

---

## 5. The Vanishing (and Exploding) Gradient Problem — derived, not just asserted

**This is THE single most important concept in this entire file, and the direct motivation for LSTM (Part 2).**

**Setting up the derivation:** because of the chain rule (section 4), computing the gradient of the loss with respect to something that happened FAR in the past (say, $h_1$, if we're currently processing $h_{100}$) requires MULTIPLYING TOGETHER approximately 99 separate "how does $h_t$ affect $h_{t+1}$" derivative terms in a row (one for each step of backward travel through the chain). **Each individual one of these terms involves the weight matrix $W_{hh}$ (the same recurrent weight matrix reused at every single time step — remember, "recurrent" specifically means the exact SAME weights are reused over and over) MULTIPLIED by the derivative of the $\tanh$ activation function at that step.**

**The key mathematical fact that causes the problem:** the derivative of $\tanh$ is ALWAYS a number between 0 and 1 (specifically, $\tanh'(z) = 1-\tanh^2(z)$, and since $\tanh(z)$ itself is always between $-1$ and $1$, this derivative is always between 0 and 1, typically noticeably LESS than 1 except very close to $z=0$). **So this long backward chain involves repeatedly multiplying together roughly 99 numbers, each one typically somewhat LESS than 1** (from the $\tanh$ derivative alone, even setting aside $W_{hh}$'s own scale for a moment). **Multiplying many numbers together that are each less than 1 produces a result that shrinks EXPONENTIALLY toward zero as the CHAIN LENGTH grows** — EXACTLY the same "geometric decay" mathematical mechanism you've now derived multiple times throughout this course (Phase 6, Part 1, section 2's AR(1) unrolling with $\phi^j$ terms; Phase 5, section 4's SES weight decay with $(1-\alpha)^j$ terms) — **here, that same geometric-decay mathematics is happening to the GRADIENT itself, rather than to a forecast or a smoothing weight.**

**The practical, genuinely serious consequence: gradients corresponding to LONG-AGO time steps become vanishingly small — meaning the network essentially receives NO learning signal telling it "you should have paid more attention to something that happened 50+ steps ago" — the network effectively becomes UNABLE to learn genuinely LONG-RANGE dependencies, no matter how many training examples you give it, because the mathematical signal needed to learn that long-range relationship has been crushed down to numerically indistinguishable-from-zero by the time it propagates backward that far.** **This is called the vanishing gradient problem**, and it's a genuine, well-documented, formula-level limitation of the vanilla RNN architecture just derived above — not a minor implementation detail, but a structural mathematical consequence of the repeated multiplication inherent in backpropagation through many time steps.

**The mirror-image problem, briefly: exploding gradients.** If, instead, the relevant terms in that repeated multiplication happen to be consistently GREATER than 1 (which CAN happen depending on the specific values $W_{hh}$ takes on), the same repeated-multiplication mechanism instead causes the gradient to grow EXPONENTIALLY LARGE rather than vanish — causing wildly unstable, erratic training updates. **In practice, vanishing gradients are the far more common and more damaging problem for standard RNNs** (exploding gradients have a fairly simple practical fix called "gradient clipping" — just capping the gradient's size if it gets too large — whereas vanishing gradients require a more fundamental architectural fix, which is exactly what Part 2's LSTM provides).

**Connecting this directly back to something you already fully derived, for a genuinely satisfying "aha":** recall Phase 6, Part 1, section 2 — an AR(1) process's own memory ALSO decays geometrically, at rate $\phi^j$, and you showed this is EXACTLY why AR(1) forgets the distant past (its own influence shrinks toward zero as lag grows). **The vanishing gradient problem is, in a very real structural sense, the LEARNING-side mirror image of that exact same geometric-decay phenomenon** — just as an AR(1) process's memory of the distant past fades away in its FORECASTS, a vanilla RNN's ability to LEARN FROM the distant past fades away in its GRADIENTS, via a structurally similar repeated-multiplication mechanism. **This is a genuinely deep, recurring pattern across this entire course (geometric decay showing up again and again in different guises), and recognizing it here is a strong sign of genuine, connected understanding rather than memorized, disconnected facts.**

---

## 6. A small numerical taste of the vanishing gradient, made concrete

Suppose, for simplicity, that at every time step, the relevant chain-rule term (combining $W_{hh}$'s effect and the $\tanh$ derivative) works out to approximately $0.5$ (a plausible, realistic-ish magnitude). **After just 10 time steps back, the CUMULATIVE gradient-multiplier is $0.5^{10} \approx 0.001$ — the original gradient signal has already shrunk to about one-tenth of one percent of its starting size.** **After 20 steps back: $0.5^{20}\approx 0.00000095$ — utterly negligible, completely swamped by ordinary numerical noise in any real computation.** **Compare this directly to Phase 6, Part 1, section 6's numerical AR(1) example, where we computed $\phi^4=0.1296$ for $\phi=0.6$ — already noticeably small after just 4 steps; here, with a comparable decay rate but MANY MORE steps typically needed for real sequences (dozens, sometimes hundreds of time steps), the SAME geometric-decay mathematics produces an even more severe, more completely negligible effect.** This concretely illustrates why a vanilla RNN genuinely cannot learn a dependency spanning even a modest number of steps, let alone the hundreds of steps genuinely relevant to, say, yearly seasonality in daily data.

---

## 7. Quick self-check questions

1. Why is the $\tanh$ activation function genuinely necessary in the RNN hidden-state equation, rather than just adding the weighted terms directly?
   *(Answer: without a non-linear activation function like tanh, the entire recurrent network — no matter how many layers or time steps — would mathematically collapse into being equivalent to a single plain linear model, exactly as limited as AR(p); tanh injects the non-linearity needed to capture genuinely non-linear relationships that linear models like AR(p) or GARCH cannot represent.)*
2. In your own words, why does computing the gradient for a time step far in the past require multiplying together many terms, and why does that cause a problem?
   *(Answer: because of the chain rule, the influence of a distant past hidden state on the current loss has to be traced backward through every intermediate time step, multiplying together one "how does h_t affect h_{t+1}" derivative term per step; since each of these terms typically has magnitude less than 1 (from the tanh derivative and the recurrent weight matrix), multiplying many of them together causes the overall gradient to shrink exponentially/geometrically toward zero as the number of steps grows, eventually becoming too small to provide any meaningful learning signal about long-past events.)*
3. How does the vanishing gradient problem in RNNs structurally parallel something you derived much earlier in this course, specifically in Phase 6, Part 1?
   *(Answer: it parallels the geometric decay of an AR(1) process's own memory, ρ(k)=φ^k — both phenomena arise from repeatedly multiplying together a factor with magnitude less than 1, causing exponential/geometric decay; AR(1)'s forecasts forget the distant past at rate φ^j, while an RNN's gradients forget how to LEARN from the distant past via a structurally similar repeated-multiplication mechanism during backpropagation.)*
4. What's the difference between the vanishing and exploding gradient problems, and which one is generally considered the more fundamentally damaging issue for standard RNNs?
   *(Answer: vanishing gradients shrink exponentially toward zero as they propagate backward through many time steps, preventing the network from learning long-range dependencies; exploding gradients instead grow exponentially large, causing unstable training. Vanishing gradients are generally considered the more fundamentally damaging problem, since exploding gradients have a relatively simple practical fix (gradient clipping), while vanishing gradients require a genuine architectural redesign — exactly what LSTM provides.)*

---

## What's next
**Part 2 of Phase 16** derives the **LSTM (Long Short-Term Memory)** architecture, gate by gate, specifically engineered to solve the vanishing gradient problem just derived — we'll build the forget gate, input gate, output gate, and cell state equations one at a time, with the same "define every symbol, explain every piece's plain-English purpose" treatment, and explain precisely WHY the LSTM's specific structure avoids the repeated-multiplication decay problem that plagues the vanilla RNN.

Say "next" for Part 2, or ask for more drilling on the vanishing gradient derivation first — genuinely worth being solid on, since Part 2's entire motivation depends on it.
