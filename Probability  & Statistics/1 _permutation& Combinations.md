# Permutations & Combinations — When To Use Which


## The One Question That Decides Everything

Before reaching for any formula, ask yourself one thing:

> **"If I rearrange my selection, do I get a different answer?"**

- **YES** — the arrangement matters → **Permutation**
- **NO** — the arrangement doesn't matter → **Combination**

That's it. Everything else follows from this.

---

## Making It Stick: The Committee vs. The Podium

Here is the cleanest example of the difference:

**Situation:** You have 10 people. You need to pick 3.

*Case A — A committee of 3:*
Picking Alice, Bob, Carol is the same as picking Carol, Bob, Alice.
The group is the group. Order doesn't matter.
→ **Combination**

*Case B — A podium (1st, 2nd, 3rd place):*
Alice 1st, Bob 2nd, Carol 3rd is **different** from Carol 1st, Alice 2nd, Bob 3rd.
The positions matter. Order matters.
→ **Permutation**

Same 10 people. Same 3 chosen. Different question = different formula.

---

## The Mental Test (Use This Every Time)

When you read a problem, mentally swap two of your chosen items and ask:

> **"Did anything change?"**

| You swap two people and... | Formula |
|---|---|
| The result is still valid and identical | **Combination** $\binom{n}{r}$ |
| The result is different (different rank, role, position) | **Permutation** $P(n,r)$ |

---

## Trigger Words

You don't always have time to think from scratch. Certain words in a problem signal which formula to use:

| These words → Permutation | These words → Combination |
|---|---|
| arrange, order, rank | choose, select, pick |
| first / second / third | group, team, committee |
| assign roles (president, VP) | how many ways to form... |
| schedule, sequence | how many possible hands/sets |
| password, PIN, code | subset, sample |

**Caution:** The word "ways" appears in both — it's not a trigger on its own. Look for *role* or *rank* language instead.

---

## The Formulas (and why they look the way they do)

$$P(n, r) = \frac{n!}{(n-r)!}$$

$$C(n, r) = \binom{n}{r} = \frac{n!}{r!\,(n-r)!}$$

The only difference between them is that $\binom{n}{r}$ has an extra $r!$ in the denominator.

**Why?** Because $r!$ is the number of ways to rearrange your chosen items. Combination divides it out — it collapses all rearrangements into one. Permutation keeps them all.

$$\binom{n}{r} = \frac{P(n,r)}{r!}$$

If you ever blank on the combination formula, just remember: *start with the permutation formula, then divide by $r!$ to remove the ordering.*

---

## Common Problems — Classified

Work through each one. Cover the answer, decide for yourself, then check.

| Problem | Order matter? | Formula | Answer |
|---|---|---|---|
| Arrange 5 books on a shelf | ✅ Yes — position matters | $P(5,5) = 5!$ | 120 |
| Choose 3 students for a trip from 10 | ❌ No — just a group | $\binom{10}{3}$ | 120 |
| Assign gold, silver, bronze to 3 of 10 athletes | ✅ Yes — medals are different | $P(10,3)$ | 720 |
| Pick a 4-person committee from 12 | ❌ No — all members equal | $\binom{12}{4}$ | 495 |
| 4-digit PIN from digits 1–9, no repeats | ✅ Yes — 1234 ≠ 4321 | $P(9,4)$ | 3024 |
| Deal a 5-card hand from 52 cards | ❌ No — a hand is a set | $\binom{52}{5}$ | 2,598,960 |
| Seat 6 people in 6 chairs | ✅ Yes — who sits where matters | $6!$ | 720 |
| Pick 2 fruits from {apple, banana, mango, orange} | ❌ No — just which two | $\binom{4}{2}$ | 6 |
| Elect a president and VP from 8 people | ✅ Yes — different roles | $P(8,2)$ | 56 |
| Form a group of 2 from those same 8 | ❌ No — no roles | $\binom{8}{2}$ | 28 |

Notice the last two: same numbers, same pool of people, different question → different answer.

---

## The Most Common Mistake

**Treating a ranked/role selection as a combination.**

> "Choose 3 winners from 10" — if those winners get gold/silver/bronze, it's a permutation.
> "Choose 3 winners from 10" — if they all just win equally, it's a combination.

The word "choose" doesn't automatically mean combination. Ask: *do the positions or roles differ?*

---

## Rule of Product — When Everything Else Is Just Multiplication

The rule of product is not really a separate formula. It's the underlying logic that *builds* permutations and combinations.

> **If you make several independent choices in sequence, multiply the number of options at each step.**

3 shirts × 4 pants = 12 outfits.
$26^3 \times 10^2$ = number of passwords with 3 letters then 2 digits.

Use it directly when you're filling slots with independent choices. Use permutation/combination when you're selecting from a pool.

| Situation | What to do |
|---|---|
| Filling independent slots (passwords, sequences) | Rule of product: multiply options per slot |
| Selecting an ordered subset from a pool | Permutation $P(n,r)$ |
| Selecting an unordered subset from a pool | Combination $\binom{n}{r}$ |

---

## One-Page Summary

```
PROBLEM SAYS...              → USE

"arrange", "order", "rank"   → P(n,r)
"first/second/third place"   → P(n,r)
"assign roles"               → P(n,r)
"password / PIN / code"      → P(n,r)  [if no repeats]
                                n^r     [if repeats allowed]

"choose", "select", "pick"   → C(n,r)
"committee / team / group"   → C(n,r)
"hand of cards"              → C(n,r)

independent slots             → multiply options per slot

WHEN IN DOUBT:
Swap two items in your selection.
Did the result change? → Permutation.
Still the same? → Combination.
```

---

*If you remember nothing else: swap two items. Did it change? Permutation. Still the same? Combination.*
