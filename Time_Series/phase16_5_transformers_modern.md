# Phase 16, Part 5 of 5: Transformers, N-BEATS, DeepAR, TFT

Roadmap: 16.1 Why RNNs + vanishing gradient [done] → 16.2 LSTM [done] → 16.3 GRU + seq2seq [done] → 16.4 TCN + Attention [done] → **16.5 Transformers + modern architectures (this file, final part)**.

Part 4 built attention as a fix for the encoder-decoder bottleneck. This file generalizes attention into the full Transformer architecture, addresses the one thing Transformers structurally lack (a sense of order), and then tours the four named architectures a modern interview might expect you to recognize by name and describe accurately.

---

## 1. Symbol glossary for this file

| Symbol | Plain-English meaning |
|---|---|
| self-attention | attention applied WITHIN one sequence — each position attends to every other position in the SAME sequence |
| positional encoding | a technique for injecting order/position information into a Transformer, since it has none built in otherwise |
| multi-head attention | running several attention computations in parallel, each potentially learning to focus on different kinds of relationships |
| basis expansion | N-BEATS' specific technique for representing a forecast, explained in section 4 |
| autoregressive (in DeepAR's sense) | generating a forecast one step at a time, feeding each output back in — the SAME recursive idea from Phase 6/Part 3, now applied inside a neural likelihood model |
| VSN | Variable Selection Network — a TFT-specific component, explained in section 6 |

---

## 2. From attention to self-attention: the key generalization behind Transformers

**Recall Part 4, section 5: attention let a DECODER position look back at ENCODER positions.** **Self-attention generalizes this by applying the EXACT SAME query/key/value mechanism WITHIN a single sequence — every position attends to EVERY OTHER position in that SAME sequence (including itself), not just across an encoder/decoder divide.** **Plain English: instead of "the output looks back at the input," self-attention asks "for each individual point in this ONE sequence, which OTHER points in that SAME sequence are most relevant to understanding/representing THIS point?"** — e.g., in a daily sales series, a self-attention layer might learn that TODAY's value should pay particular attention to the SAME DAY LAST WEEK, and the SAME DAY LAST YEAR, directly — regardless of how far away those points are in raw step-count, since self-attention (like Part 4's attention) computes a DIRECT connection between any two positions, without needing information to travel sequentially through every point in between.

**The Transformer architecture, in one plain-English sentence: it's built ENTIRELY out of stacked self-attention layers (plus simple feed-forward/regression-style layers in between), with NO recurrence (Parts 1-3) and NO convolution (Part 4's TCN) anywhere in the architecture at all.** **This is a genuinely radical structural choice, and it has a direct, important consequence, addressed in the next section.**

**Multi-head attention, briefly:** rather than computing just ONE single set of Query/Key/Value projections (Part 4, section 5), a Transformer computes SEVERAL separate sets IN PARALLEL (called "heads") — **plain English: each head can potentially learn to specialize in a DIFFERENT KIND of relationship** (e.g., one head might learn to focus on short-term, adjacent dependencies; another might specialize in long-range, same-day-last-year style dependencies) **— then the results from all heads are combined together.** This is directly analogous, in spirit, to how an ENSEMBLE of models (Phase 15, section 6) can capture different kinds of patterns better than any single model alone — here, multiple attention "heads" within ONE layer play a similar diversifying role.

---

## 3. Positional Encoding: the problem Transformers create for themselves, and the fix

**The genuinely important structural problem, worth deriving precisely, not just asserting:** self-attention (section 2) computes relevance between EVERY PAIR of positions using ONLY their content (Query and Key vectors, Part 4, section 5) — **notice, critically, that NOTHING in the raw self-attention computation itself references WHERE in the sequence each position actually sits.** **If you took an input sequence and completely SHUFFLED/REORDERED its positions, a pure self-attention layer would compute the EXACT SAME set of pairwise relevance scores between any two given CONTENT values, regardless of their actual original ORDER — self-attention is, by its raw mathematical construction, entirely "order-blind."** **This is a genuinely serious problem for TIME SERIES specifically, since — as established all the way back in Phase 1, section 1 — ORDER IS THE DEFINING FEATURE of what makes something a time series at all; a model that's fundamentally blind to order cannot, on its own, distinguish "this happened before that" from "that happened before this."**

**The fix: positional encoding — deliberately, explicitly ADD information about EACH position's location in the sequence directly INTO its input representation, before any self-attention computation happens.** **The standard technique (the original Transformer paper's approach): use sine and cosine functions of DIFFERENT frequencies, evaluated at each position — genuinely, directly the SAME Fourier machinery from Phase 8, section 3, and Phase 14, section 4, just repurposed here for a new goal.**
$$
PE_{(pos, 2k)} = \sin\left(\frac{pos}{10000^{2k/d}}\right), \qquad PE_{(pos,2k+1)}=\cos\left(\frac{pos}{10000^{2k/d}}\right)
$$
**Plain English, without getting lost in the specific constant 10000 or dimension $d$ (implementation details you don't need to memorize): at each position (`pos`) in the sequence, compute a WHOLE SET of sine/cosine values, each pair evaluated at a DIFFERENT frequency (some fast-oscillating, some slow-oscillating, echoing Phase 8, section 3's multiple-$k$ idea) — this produces a unique, position-specific "fingerprint" pattern of numbers for EVERY position in the sequence, which then gets ADDED directly to that position's content representation before self-attention ever runs.** **Because each position gets a genuinely DIFFERENT, systematically-varying fingerprint (thanks to using MULTIPLE different frequencies simultaneously, exactly as Phase 8 needed multiple $k$ values to build flexible, detailed seasonal shapes), the self-attention mechanism CAN now, in principle, learn to distinguish and make use of relative or absolute position information — even though the core self-attention computation itself remains fundamentally order-blind, the INPUT it receives is no longer order-blind, since position has been explicitly baked in beforehand.**

**A genuinely satisfying, complete interview-ready answer to "why do Transformers need positional encoding, and how does it work": "unlike RNNs, which process a sequence strictly in order and therefore have an inherent sense of position built into their sequential structure, self-attention computes relevance between all pairs of positions purely from content, with no reference to their actual order — making it fundamentally order-blind by construction. Positional encoding fixes this by adding a unique, sine/cosine-based fingerprint to each position's representation before self-attention runs, giving the model the information it needs to distinguish position and order, using the same multi-frequency Fourier-style approach used elsewhere in this course for representing seasonal patterns."**

---

## 4. N-BEATS / N-HiTS: forecasting via learned basis expansion

**The core idea, in plain English:** rather than using recurrence (Parts 1-3), convolution (Part 4), or self-attention (sections 2-3), **N-BEATS is built from stacked blocks of simple, ordinary feed-forward (fully-connected, non-recurrent, non-convolutional) neural network layers — but with a genuinely clever OUTPUT structure specifically designed for interpretability and forecasting.**

**New term: basis expansion.** Plain English, connecting directly to something you already fully understand: **recall Phase 8's Fourier terms — representing a complicated seasonal SHAPE as a weighted combination of simple building-block waves (sines and cosines).** **N-BEATS generalizes this EXACT idea: instead of PRE-SPECIFYING the building blocks (as Fourier terms do, always using sine/cosine), each block in N-BEATS LEARNS its OWN set of building-block "basis functions" directly from the data, and produces a forecast as a WEIGHTED COMBINATION of these LEARNED basis functions** — the SAME fundamental "build something complex from a weighted sum of simpler building blocks" philosophy as Phase 8's Fourier approach, just with the building blocks themselves being LEARNED rather than fixed in advance as pure sine/cosine waves.

**A genuinely distinctive, useful design feature: each N-BEATS block outputs BOTH a forecast contribution AND a "backcast" (a reconstruction of the INPUT it was given) — the backcast's job is to explain away/reconstruct whatever part of the input this particular block has already accounted for, and the RESIDUAL (whatever the backcast couldn't reconstruct) gets passed on to the NEXT block in the stack, which then focuses on explaining THAT leftover residual.** **This should feel structurally familiar: it's directly analogous to gradient boosting's core idea (Phase 15, section 2 — each new tree specifically targets the RESIDUAL errors left over by all previous trees) — N-BEATS applies a very similar "each new component targets what's still unexplained" philosophy, just using neural network blocks instead of decision trees.**

**N-HiTS (a more recent, closely related extension), briefly:** adds hierarchical, multi-rate structure — different blocks specifically specialize in different TIME SCALES (some blocks focus on fine-grained, short-term patterns; others focus on smoothed, longer-term patterns) — directly echoing the "multiple overlapping seasonalities handled via separate blocks" idea from Phase 8, section 4, just now applied to general trend/pattern components rather than specifically to seasonal Fourier terms.

---

## 5. DeepAR: a probabilistic, autoregressive RNN

**The core idea, connecting directly back to Part 1-2's RNN/LSTM machinery, PLUS Phase 6's probabilistic forecasting ideas:** DeepAR (developed at Amazon) uses an ordinary RNN/LSTM (Parts 1-2) to process the sequence — **but instead of directly outputting a single point forecast number, at EACH time step it outputs the PARAMETERS of a probability distribution** (e.g., for a Normal distribution, it outputs a predicted MEAN and a predicted VARIANCE/standard deviation, at every single time step — directly generalizing Phase 6, Part 4's MLE framework, but now with the distribution's parameters THEMSELVES being dynamically predicted by a neural network at every step, rather than being fixed, single, constant values estimated once for the whole series the way plain ARMA's $\sigma^2$ was).

**"Autoregressive" in DeepAR's specific sense (worth being precise about, since this term gets reused across different contexts, similar to the various Greek-letter reuse throughout this course):** during FORECASTING (not training), DeepAR generates each future step by SAMPLING from its predicted distribution at that step, then FEEDS THAT SAMPLED VALUE back in as input for predicting the NEXT step — **this is EXACTLY the same "feed your own output back in as the next input" recursive logic from Phase 6, Part 5, section 4.2, and Part 3, section 3's decoder — just now generating an entire probability distribution and SAMPLING from it at each step, rather than deterministically feeding forward a single point forecast.**

**A genuinely important practical advantage, directly connecting to Phase 15, section 4's global-model discussion: DeepAR is specifically designed as a GLOBAL model** — trained on MANY related series simultaneously (exactly Phase 15's pooling strategy), which is precisely how it handles the "cold start" problem for new series with little individual history — **it was specifically built by Amazon for exactly this kind of large-scale, many-related-series forecasting problem (e.g., forecasting demand for thousands of different products simultaneously), directly the kind of scenario flagged as a realistic Google/Apple interview scenario throughout this course.**

**Because DeepAR outputs a full PROBABILITY DISTRIBUTION at every step (not just a point estimate), you can naturally generate MULTIPLE possible future trajectories by repeatedly sampling — producing genuine prediction intervals and full probabilistic forecasts DIRECTLY, connecting cleanly back to Phase 13, sections 8-9's pinball loss and CRPS evaluation metrics, which are precisely the RIGHT tools for evaluating exactly this kind of full-distribution output.**

---

## 6. Temporal Fusion Transformer (TFT): combining nearly everything from this entire phase

**Plain English framing: TFT is a genuinely comprehensive, modern architecture that specifically combines LSTM (Parts 1-2, for local, short-term sequential processing), self-attention (sections 2-3, for long-range dependencies), AND several specifically business/interpretability-focused components — designed to be both highly accurate AND genuinely interpretable (a real, practical concern for production forecasting systems that need to be trusted and explained to business stakeholders, not just accurate on paper).**

**Key distinctive components, each explained briefly:**

**Variable Selection Networks (VSNs):** a learned mechanism that automatically decides, for EACH INDIVIDUAL input feature (recall Phase 14's whole toolkit — lags, rolling stats, calendar features, holiday flags), HOW MUCH that particular feature should matter for the current prediction — genuinely directly analogous, in spirit, to attention's "how much should I focus on this" weighting logic (section 2), but applied across DIFFERENT FEATURES rather than across different TIME POSITIONS. **Plain English: rather than treating all of Phase 14's engineered features as equally important always, TFT learns to dynamically emphasize whichever specific features are most relevant right now, and can reveal (for interpretability) WHICH features it's currently prioritizing.**

**A mix of LSTM layers (for capturing short-term, local sequential patterns, Parts 1-2) AND self-attention layers (for capturing long-range dependencies, sections 2-3)** — **a genuinely sensible, practical hybrid design: use the architecture best suited for each specific kind of pattern, rather than committing entirely to one single architectural philosophy.**

**Quantile outputs:** directly connecting to Phase 13, section 8's pinball loss — TFT is typically trained to output MULTIPLE specific quantiles (e.g., the 10th, 50th, and 90th percentiles) simultaneously, giving genuine, calibrated prediction intervals directly, evaluated using exactly the pinball loss machinery you already fully derived.

**Built-in interpretability outputs:** TFT can directly reveal WHICH input features it's weighting most heavily (via the VSN) and WHICH time steps its attention layers are focusing on — a genuinely important, real practical advantage for production deployment, where being able to EXPLAIN a forecast (e.g., "the model is currently weighting the upcoming holiday heavily") is often nearly as important as the forecast's raw accuracy, for building organizational trust in an automated forecasting system.

---

## 7. A concise, interview-ready summary table of all four architectures

| Architecture | Core building block | Genuinely distinctive feature |
|---|---|---|
| Transformer (general) | Self-attention, stacked | Order-blind by construction; needs positional encoding; highly parallelizable |
| N-BEATS/N-HiTS | Stacked feed-forward blocks | Learned basis expansion + backcast-residual chaining (boosting-flavored) |
| DeepAR | RNN/LSTM | Outputs full probability distributions at each step; global model; genuinely probabilistic via sampling |
| TFT | LSTM + self-attention + VSN | Combines local (LSTM) and long-range (attention) patterns; built-in feature-level and time-level interpretability; native quantile outputs |

---

## 8. Quick self-check questions

1. Precisely why is raw self-attention, by its mathematical construction, "order-blind," and how does positional encoding fix this?
   *(Answer: self-attention computes relevance scores between positions using only their Query/Key content vectors, with nothing in that computation referencing each position's actual location in the sequence — shuffling the input would produce identical pairwise relevance scores between any two given content values. Positional encoding fixes this by adding a unique, sine/cosine-based (multi-frequency) fingerprint to each position's representation BEFORE self-attention runs, giving the model the information it needs to distinguish and use position/order.)*
2. How is N-BEATS' "backcast and residual" chaining across blocks conceptually similar to gradient boosting from Phase 15?
   *(Answer: each N-BEATS block reconstructs (backcasts) whatever part of the input it can explain, and passes the leftover, unexplained residual to the next block, which then focuses specifically on that remaining residual — directly paralleling how each new tree in gradient boosting specifically targets the residual errors left over by all previous trees.)*
3. In what specific sense is DeepAR "autoregressive," and what does it output at each step that a plain point-forecasting RNN does not?
   *(Answer: during forecasting, DeepAR generates each future value by sampling from its predicted probability distribution and feeding that sampled value back in as the input for predicting the next step — the same "feed your own output back in" recursive logic as classical multi-step forecasting; unlike a plain point-forecasting RNN, DeepAR outputs the PARAMETERS of a full probability distribution (e.g., mean and variance) at every step, rather than a single deterministic number.)*
4. What are the two structurally different types of pattern-capturing components combined inside TFT, and what does the Variable Selection Network specifically do?
   *(Answer: TFT combines LSTM layers (for local, short-term sequential patterns) with self-attention layers (for long-range dependencies); the Variable Selection Network learns, for each individual input feature, how much that feature should matter for the current prediction — dynamically and interpretably weighting features rather than treating them as equally important at all times.)*

---

## Phase 16 complete — full recap

Across five parts, you built: the motivation for recurrent processing and the precise, derived reason vanilla RNNs fail on long sequences (vanishing gradients, Part 1); LSTM's gate-by-gate fix, with a genuine structural explanation for WHY it works (additive, gated memory instead of forced multiply-and-squash, Part 2); GRU as a principled simplification, and sequence-to-sequence architectures for genuine multi-step forecasting (Part 3); TCN's exponentially-efficient dilated convolutions and the attention mechanism that fixes the encoder-decoder bottleneck (Part 4); and finally, self-attention, positional encoding, and a tour of four genuinely important modern named architectures (Part 5, this file). **This is a complete, first-principles deep learning foundation for time series — the material most likely to distinguish a strong candidate in a senior-level Google/Apple forecasting interview, since most candidates can name these architectures but comparatively few can derive WHY each one works the way it does.**

## What's next
Phase 17 moves into **Anomaly and Change Point Detection** — statistical control charts, CUSUM, STL-residual-based anomaly detection, and the specific techniques (like Twitter's Seasonal Hybrid ESD) used to automatically flag unusual points in a time series — genuinely practical, widely-applicable material building directly on Phase 6, Part 5's residual diagnostics and Phase 5's STL decomposition.

Say "next" for Phase 17, or ask for more drilling on any part of Phase 16 first — or, given how much ground Phase 16 covered, feel free to ask for a consolidated Phase 16 cheat-sheet pulling every architecture's key formula and defining insight onto one page.
