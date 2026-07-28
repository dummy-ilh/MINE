# Agents — Interview Question Bank (Practice Set)

This is a standalone set of questions you'd realistically get asked, organized by interview-round type. Answers aren't written out here on purpose — use it to self-test against Modules 1-9 and the LangChain notes; where you blank, that's your signal for what to re-read.

---

## Round type 1: Conceptual/definitional (warm-up, first 10 min)

1. What's the difference between a single LLM call and an agent?
2. What does "agentic" actually mean, architecturally?
3. Explain the ReAct loop and name exactly which parts are model-generated vs. framework-generated.
4. What's the difference between chain-of-thought prompting and ReAct?
5. Name the four types of agent memory and give a one-sentence definition of each.
6. What's the difference between episodic and semantic memory, with an example?
7. What is a tool/function call, mechanically — what does the model actually output?
8. Why can't the model directly execute code or call an API itself?
9. What is Tree-of-Thought, and what problem does it solve that ReAct doesn't?
10. What's the difference between MCTS and plain Tree-of-Thought search?
11. What is Reflexion, and why is it called "verbal reinforcement learning"?
12. Name three multi-agent architecture patterns and describe each in one sentence.

---

## Round type 2: Mechanism/derivation (whiteboard-style, mid interview)

13. Walk through, step by step, what happens when an agent decides to call a tool — from the model's generation to the result coming back into context.
14. Explain mechanistically (not just "it works empirically") why chain-of-thought improves accuracy on multi-step problems.
15. Derive/explain the compounding-error math for an N-step agent task with per-step success probability p. Then explain why this matters for evaluation.
16. Write the UCB1 formula used in MCTS and explain what each term does. Walk through a numeric example where the lower-average node still gets selected.
17. Explain self-consistency and why majority voting across sampled reasoning chains improves accuracy — what's the actual statistical argument?
18. What is constrained decoding, and why does it matter specifically for reliable tool-calling?
19. Explain least-to-most prompting and how it's meaningfully different from plain CoT.
20. In a Tree-of-Thought search, name and explain the four components the technique formalizes.

---

## Round type 3: Compare and decide (judgment, not recall)

21. When would you use ReAct vs. Tree-of-Thought for a given task?
22. When would you use MCTS instead of plain ToT's BFS/DFS?
23. When is it worth splitting into multiple agents instead of using one well-designed agent?
24. Orchestrator-worker vs. debate vs. sequential pipeline — when would you pick each?
25. When would you add Reflexion vs. Tree-of-Thought to an underperforming agent?
26. Shared/global context vs. private/filtered context across multiple agents — what's the tradeoff, and how would you decide?
27. Structured vs. free-form communication between agents in a multi-agent system — when does each make sense?

---

## Round type 4: Diagnose the failure (the bulk of real interview time)

28. Your agent keeps calling the same tool with slightly different but equally wrong arguments and never makes progress. What's happening, and how do you fix it?
29. A production agent that calls a payment tool occasionally double-charges users after a timeout. Diagnose and fix.
30. A multi-agent pipeline's final report contains a factual error that no individual worker's output contained on its own. Diagnose.
31. Your agent's measured task success rate looks great in testing but users report frequent real-world failures. What's the gap, and how do you investigate?
32. An agent occasionally generates a fake "Observation" instead of actually calling the tool. What's the root cause and the fix?
33. Two debating agents in an adversarial architecture keep going back and forth without ever converging on an answer. Diagnose and propose a fix.
34. Your agent has 95% per-call tool accuracy, but overall multi-step task success is much lower than you'd expect. Explain why, with numbers.
35. An agent's memory retrieval keeps surfacing irrelevant past episodes, diluting its context. What's going wrong and how would you improve it?

---

## Round type 5: System design (30-45 min, most senior-signal round)

36. Design an agent that can book a flight for a user given natural-language preferences. Walk through the full architecture, tools, memory, and evaluation plan.
37. Design a customer-support agent that needs to look up account info, issue refunds, and escalate to a human when uncertain. What guardrails do you build in?
38. Design an evaluation pipeline for a coding agent that fixes GitHub issues. What metrics, what environment, how many trials, and why?
39. Design a multi-agent research assistant that gathers information from multiple sources and writes a synthesized report. What architecture, and how do you prevent compounding errors in the synthesis step?
40. You're told to add long-term memory to an existing single-turn chatbot to make it feel more personalized across sessions. What would you actually build?
41. Design an agent for a task with a large branching decision space (e.g., a strategic game or complex scheduling problem). Would you use ReAct, ToT, or MCTS, and why?

---

## Round type 6: Practical/tooling

42. If asked to actually build an agent for a take-home or demo, what would you reach for and why (LangChain/LangGraph, a custom loop, something else)?
43. How would you debug why a specific agent run failed, in practice?
44. How would you set up regression testing so that a prompt or tool change doesn't silently make the agent worse?
45. What's the difference between LangChain and LangGraph, and when would you need the latter?

---

## Round type 7: Cross-topic connections (signals deep understanding, not memorization)

46. How does the compounding-error math you'd use for agent evaluation relate to the "emergent abilities" debate from LLM Basics?
47. How is self-consistency (Agents) mathematically related to MCTS's simulation-averaging?
48. Are agents typically trained with RL? Explain precisely what is and isn't happening with the underlying model weights in a typical ReAct-based agent.
49. How does KL divergence show up in both RLHF (LLM Basics) and any part of the Agents material you've covered — or does it not?
50. Constrained decoding for tool-calling reuses which LLM Basics concept, and how exactly?

---

## How to use this

Go through in order once, cold, no notes — mark anything you hesitate on. Then re-read the specific module that question maps to (most map cleanly to one Agents module; a few in Round 7 pull from LLM Basics too). Repeat the ones you missed after a day or two — the diagnose-the-failure and system-design rounds are where real interviews actually spend most of their time, so weight your review there if you're short on time.
