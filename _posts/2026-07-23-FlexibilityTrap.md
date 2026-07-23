---
title: "[Paper Review] The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models"
date: 2026-07-23
categories:
  - paper review
tags:
  - Diffusion Language Models
  - Reinforcement Learning
  - GRPO
  - Reasoning
use_math: true
classes: wide
---

![FlexibilityTrap]({{ site.baseurl }}/assets/FlexibilityTrap/FlexibilityTrap_main.png)

## Introduction

In this post, I review **The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models**, a collaboration between **Tsinghua University (LeapLab)** and **Alibaba Group**, and a recipient of an **ICML 2026 Outstanding Paper Award**.

Diffusion Language Models (dLLMs) are usually sold on one headline feature: unlike autoregressive (AR) models that must generate strictly left-to-right, a dLLM can fill in tokens in **any order** it likes. The community has treated this order flexibility as a free lunch, useful for infilling, planning, and constraint-satisfaction puzzles like Sudoku.

**The Flexibility Trap** makes a provocative, counterintuitive claim: for general reasoning tasks like math and coding, arbitrary order is not a feature but a **trap**. Models exploit the freedom to dodge the hard, high-uncertainty tokens that actually matter for reasoning, which prematurely collapses the diversity of solutions they can reach. The authors then show that the fix is almost embarrassingly simple, a minimalist RL recipe they call **JustGRPO**.

---

## Paper Info

- **Title**: The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models
- **Authors**: Zanlin Ni, Shenzhi Wang, Yang Yue, Tianyu Yu, Weilin Zhao, Yeguo Hua, Tianyi Chen, Jun Song, Cheng Yu, Bo Zheng, Gao Huang
- **Conference**: ICML 2026 (Outstanding Paper Award)
- **Paper Link**: [ArXiv](https://arxiv.org/abs/2601.15165)
- **Project Page**: [nzl-thu.github.io/the-flexibility-trap](https://nzl-thu.github.io/the-flexibility-trap/)

---

## Background: Autoregressive vs. Diffusion Language Models

To understand why "order" is even a design choice, we need to contrast the two paradigms for generating text.

### Autoregressive (AR) Models
This is the family behind GPT, Llama, and virtually every mainstream LLM. An AR model factorizes the probability of a sequence $$x = (x_1, \dots, x_L)$$ using the **chain rule**, strictly left-to-right:

$$p_\theta(x) = \prod_{i=1}^{L} p_\theta(x_i \mid x_1, \dots, x_{i-1})$$

Each token is predicted conditioned only on the tokens **before** it, so generation is inherently **sequential**: you cannot sample $$x_i$$ until $$x_{i-1}$$ exists. A causal attention mask enforces this "no peeking at the future" rule during both training and inference.

* **Pros:** stable training (simple next-token objective), a well-defined per-token likelihood, and strong reasoning performance.
* **Cons:** decoding is one token at a time, so latency scales with sequence length. The generation *order* is fixed and non-negotiable.

### Diffusion Language Models (dLLMs)
Diffusion LLMs (e.g. LLaDA) borrow the idea of iterative denoising from image diffusion and adapt it to discrete text via **masked denoising**. Instead of growing a sequence left-to-right, a dLLM starts from a fully **masked** sequence and progressively **unmasks** tokens over several refinement steps:

$$[\texttt{MASK}, \texttt{MASK}, \dots, \texttt{MASK}] \;\longrightarrow\; \dots \;\longrightarrow\; [x_1, x_2, \dots, x_L]$$

Because it uses **bidirectional attention** (every position can attend to every other), the model can predict any masked position at any step. At each step it typically unmasks whichever tokens it is most confident about (low-confidence remasking). This buys two properties AR models lack:

* **Parallel decoding:** many tokens can be filled in a single forward pass, so a full sequence needs far fewer steps than its length, i.e. lower latency.
* **Arbitrary generation order:** the model is free to decide which tokens to commit first, rather than being forced through position 1, 2, 3, ….

### Why the Order Matters Here
That second property, **arbitrary order**, is the entire subject of this paper. It is intuitively appealing: for infilling or constraint-satisfaction puzzles (Sudoku, code with fixed signatures), being able to lock down the "easy," high-certainty cells first and reason inward is genuinely useful. The prevailing assumption was that this flexibility is always an advantage over the rigid left-to-right AR schedule.

The Flexibility Trap challenges exactly that assumption. It asks what happens when a model with a rigid AR schedule is compared, head-to-head, against the same model allowed to choose its own order, and finds that for open-ended reasoning, the freedom backfires.

---

## The Diagnosis: What Is the "Flexibility Trap"?

The paper's first half is a diagnosis, not a method. The authors ask a clean question: **does arbitrary-order decoding actually help reasoning?** To answer it, they compare two decoding regimes on the same pretrained dLLM:

* **Arbitrary Order** — the standard diffusion decoding that unmasks whichever tokens the model is most confident about (low-confidence remasking).
* **AR Order** — forcing the model to resolve the leftmost unresolved token first, left-to-right, while keeping bidirectional attention intact.

![Pass@k scaling]({{ site.baseurl }}/assets/FlexibilityTrap/FlexibilityTrap_passk.png)

### 1. Solution Coverage Collapse
The key measurement tool is **Pass@k**, which probes the upper bound of reasoning by asking whether any of $$k$$ sampled solutions is correct. Across three dLLMs and four benchmarks, arbitrary order shows **notably flatter Pass@k curves** than AR order, i.e. sampling more does not open up new solutions.

Worse, the two solution sets are not merely different, they are **nested**. On HumanEval, AR order solves **21.3%** of problems that arbitrary order never gets, while arbitrary order solves only **0.6%** that AR order misses. Arbitrary-order solutions are essentially a strict subset of AR-order ones. Flexibility does not expand the reachable solution space, it shrinks it.

### 2. The Mechanism: Entropy Degradation
Why does this happen? The authors trace it to **how each regime confronts uncertainty**:

* **AR order** must resolve the next token in sequence, so it is forced to confront high-uncertainty positions and keep multiple reasoning branches alive.
* **Arbitrary order** greedily commits the easy, high-confidence tokens first and **defers the hard, high-uncertainty ones**, especially logical connectives like "Therefore," "Thus," "Since" that act as reasoning forks.

By the time the model finally fills those pivotal tokens, the surrounding context is already locked in, so the "fork" collapses to a single, low-entropy continuation. The paper calls this **entropy degradation**: uncertainty is quietly drained at exactly the decision points where exploration mattered most.

---

## The Fix: JustGRPO

If arbitrary order sabotages exploration during learning, the remedy is to simply **not use it during training**. That is the whole idea behind **JustGRPO** ("just GRPO", no diffusion-specific machinery).

![JustGRPO]({{ site.baseurl }}/assets/FlexibilityTrap/FlexibilityTrap_method.png)

### 1. AR Trajectory as a Training Scaffold
During RL exploration, JustGRPO constrains the model to a standard left-to-right trajectory by masking all future tokens:

$$\tilde{x}_t = [\,o_1, \dots, o_{t-1}, \texttt{[MASK]}, \dots, \texttt{[MASK]}\,]$$

Because generation is now sequential, the per-token likelihood is well defined, and **standard Group Relative Policy Optimization (GRPO)** applies directly, with no trajectory approximations, no marginal-likelihood estimation, and no diffusion-tailored surrogate objective. The AR constraint is a training scaffold for clean credit assignment, nothing more.

### 2. Nothing Is Sacrificed at Inference
Crucially, the constraint is applied **only during training**:
* The architecture is **unchanged** — the model keeps its native **bidirectional attention**.
* At inference, the model recovers full **parallel decoding** using a training-free **EB-Sampler**.

In fact, the AR-trained model becomes more robust to the approximation error of parallel sampling, so throughput and accuracy improve together rather than trading off.

---

## Results

JustGRPO was evaluated on **LLaDA-Instruct** across math (GSM8K, MATH-500) and code (HumanEval, MBPP), against a battery of diffusion-specific RL methods (ESPO, GDPO, SPG, d-TreeRPO).

**Quantitative comparison (accuracy):**

| Benchmark | Length | Best Prior Method | **JustGRPO** |
| :--- | :---: | :---: | :---: |
| GSM8K | 256 | SPG: 86.1 | **89.1** |
| MATH-500 | 256 | SPG: 40.0 | **45.1** |
| HumanEval | 128 | ESPO: 28.1 | **37.8** |
| MBPP | 256 | GDPO: 50.6 | **52.4** |

**Key Findings:**
* **Simplicity wins:** plain GRPO, once decoupled from arbitrary order, **beats every specialized diffusion RL method**, most decisively on code (HumanEval +9.7 over ESPO).
* **Parallelism preserved (and improved):** with the EB-Sampler, MBPP gains **+10.6%** at 1 token/step and **+25.5%** at ~5 tokens/step, so the model reasons better and decodes faster.
* **The trap is real:** the improvements track exactly the exploration failure the diagnosis predicted, restoring the Pass@k scaling that arbitrary order had flattened.

---

## Takeaways

The Flexibility Trap is a rare "measure first, fix second" paper, and its lesson generalizes beyond dLLMs.

* **Flexibility is not free.** A capability that looks strictly additive (decode in any order) can be actively harmful if the model uses it to avoid difficulty rather than resolve it. For reasoning, confronting uncertainty head-on is the point.
* **Align the training signal, not the architecture.** The authors do not redesign the model or invent a new objective, they just remove a degree of freedom during learning so that credit assignment becomes honest. The bidirectional dLLM is fully intact at inference.
* **A caveat worth keeping.** The paper is careful: arbitrary order still genuinely helps constraint-satisfaction tasks like Sudoku. The claim is scoped to open-ended reasoning, where solution coverage, not infilling, is the bottleneck.

For anyone building reasoning-capable diffusion LLMs, the takeaway is blunt: **use order flexibility where it helps, but train left-to-right where it matters.**
