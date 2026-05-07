# How Deep Do Tree of Thoughts (ToT) Systems Go in Practice?

Excellent question — and this is actually where Tree of Thoughts becomes much closer to classical AI search algorithms than ordinary prompting.

In practice, the loop is usually:

Generate → Evaluate → Prune → Generate Deeper → Evaluate Again

repeated MANY times.

But the depth depends heavily on:
- cost
- latency
- task complexity
- branching factor
- model quality

---

# In Practice, Most ToT Systems Are Surprisingly Shallow

Usually:
- depth 2–5 is already common
- depth 10+ becomes expensive very quickly

Because the explosion is enormous.

---

# Why Trees Explode Exponentially

Suppose:
- each node generates 3 children
- and you go 5 levels deep

Then:

Level 0:
1 node

Level 1:
3 nodes

Level 2:
9 nodes

Level 3:
27 nodes

Level 4:
81 nodes

Level 5:
243 nodes

Total explored thoughts:

1 + 3 + 9 + 27 + 81 + 243 = 364

And EACH node may require:
- a generation call
- an evaluation call
- maybe tool calls
- maybe another reflection call

So costs explode extremely fast.

---

# That Is Why Pruning Is EVERYTHING

The evaluator is not just “nice to have.”

It is what prevents this:

3^d

where:
- d = tree depth

Without pruning, ToT becomes computationally impossible very quickly.

---

# Real-World Implementations

## Simple Educational Demos

Depth:
1–2

Like your code.

---

## Practical Agent Systems

Depth:
2–5

Common in:
- planning agents
- coding agents
- reasoning workflows

---

## Advanced Research Systems

Depth:
10+

But:
- heavily pruned
- beam searched
- heuristically scored
- often parallelized

Otherwise token costs become enormous.

---

# What Usually Happens Instead of “Pure Deep Trees”

Modern systems often do:

## Beam Search

Keep only top K branches.

Example:

Generate 10
    ↓
Keep best 2
    ↓
Generate deeper
    ↓
Keep best 2 again

This is VERY common.

---

# Another Important Insight

Most modern agent systems are actually hybrids of:
- ReAct
- Planning
- ToT
- Reflection
- Tool use

rather than “pure ToT.”

For example:
- Claude Code
- Devin-like systems
- OpenAI deep reasoning agents
- SWE agents

often behave more like:

Plan
  ↓
Generate branches
  ↓
Use tools
  ↓
Evaluate
  ↓
Reflect
  ↓
Revise plan
  ↓
Continue

So the tree becomes:
- dynamic
- adaptive
- partially linear
- partially branching

---

# Your Current Code

Your current code is essentially:

Depth = 2

because:
1. initial generation
2. one deeper expansion

If you added another cycle like:

```python
for promising_branch in promising_branches:
    evaluate_again()
    generate_again()