**Stack‑Augmented Decoder‑Only Transformer with Frozen Arithmetic Tools**

---

#### 1 · Research Objective  
Explore whether adding **hard, parameter‑free stack operations** to a Transformer decoder improves *compositional reasoning, arithmetic accuracy,* and *length extrapolation*—without increasing the number of trainable parameters inside the tools themselves.

---

#### 2 · Core Ideas  

| Component | Design choice | Rationale |
|-----------|---------------|-----------|
| **Functional stack** | Fixed‑depth tensor `[B, D, d_model]` plus integer pointer `[B]`. | Keeps XLA shapes static; pointer is treated as non‑diff to avoid ill‑posed gradients. |
| **Toolbox** | `PUSH(token)` and `ADD_TOP_TWO()` implemented with pure JAX/XLA ops and wrapped in `lax.stop_gradient`. | Tools are *frozen*—they expose gradients to their *inputs* but do not learn internal parameters, guaranteeing deterministic behaviour. |
| **Transformer backbone** | Standard decoder layer (`LN + Self‑Attention + MLP`) repeated *L* times. | Provides expressive sequence modelling while remaining comparable to vanilla baselines. |
| **Controller head** | Tiny 3‑way linear classifier on each layer’s summary vector → hard `argmax` via `lax.switch`. Choices: *no‑op*, *push*, *add‑then‑push*. | Lets the network learn **when** to invoke a tool but not **how** the tool works, emulating symbolic function calls. |
| **Gradient flow** | Straight‑through for the action index; full back‑prop through the backbone and controller; zero gradients inside tools. | Separates *learning to decide* from *learning the operation*, matching prior neural‑symbolic schemes. |

---

#### 3 · Implementation Snapshot (Flax + JAX)

```python
stack, ptr = jnp.zeros((B,D,d)), jnp.zeros((B,), jnp.int32)
for layer in layers:
    h = transformer_block(h)                 # usual SA+MLP
    logits = nn.Dense(3, use_bias=False)(h[:,-1])  # controller
    act = jnp.argmax(logits, -1)

    def noop(args): s,p,t = args;  return s,p
    def push(args): s,p,t = args;  return push_tool(s,p,t)
    def add_then_push(args):
        s,p,t = args
        t2 = add_tool(s,p)
        return push_tool(s,p,t2)

    stack, ptr = lax.switch(act,
        (noop, push, add_then_push),          # branches
        (stack, ptr, h[:,-1]))                # args
```

*Push* and *Add* rely only on `lax.dynamic_update_slice` and `lax.dynamic_slice`, so they JIT‑compile on GPU/TPU immediately.

---

#### 4 · Research Questions & Hypotheses  

| ID | Question | Hypothesis |
|----|----------|------------|
| **Q1** | Does the stack improve generalisation on context‑free languages (e.g. Dyck k) over a width‑matched Transformer? | Yes—explicit LIFO memory should scale to deeper nesting without training‑time exposure. |
| **Q2** | Can the network learn multi‑digit addition (e.g. bAbI “add two numbers”) with fewer parameters than a Neural Arithmetic Logic Unit baseline? | Likely—`ADD_TOP_TWO` provides an inductive bias that NALU learns implicitly. |
| **Q3** | Does freezing the tools harm task adaptation compared with jointly‑trained soft stacks? | No—separating control from computation should stabilise training and reduce interference. |

---

#### 5 · Experimental Plan  

| Domain | Dataset / Task | Metric | Baselines |
|--------|----------------|--------|-----------|
| **Formal languages** | Dyck‑2, Dyck‑3 (varying nesting depth) | accuracy @ length | Vanilla Transformer; Stack‑Attention (soft) |
| **Algorithmic** | IOClean Addition, Reversal | exact string match | Neural GPU; NALU |
| **Arithmetic QA** | GSM8K subset (add‑only) | exact answer |  Tiny‑GPT model |
| **Code reasoning** | Simple postfix‑to‑infix translation | BLEU | Seq2Seq LSTM |

Ablations: remove `ADD_TOP_TWO`; swap hard argmax for Gumbel‑Softmax; vary stack depth D.

---

#### 6 · Evaluation Protocol  

* **Length extrapolation curve**: train on sequences ≤ N, test up to 4 × N.  
* **Intervention analysis**: record controller actions; measure mutual information between “should push” oracle signal and learnt policy.  
* **Resource usage**: compare FLOPs and wall‑time to baselines with equivalent hidden size.

---

#### 7 · Expected Contributions  

1. **Architecture**: first decoder‑only Transformer that integrates a *hard, frozen, differentiable* stack and arithmetic operator at every layer.  
2. **Empirical evidence** that frozen symbolic tools can beat learned soft memories on depth‑extrapolation tasks with lower parameter count.  
3. **Open‑source JAX/Flax reference** suitable for further research on modular neural‑symbolic systems.

---

#### 8 · Risks & Mitigations  

| Risk | Mitigation |
|------|------------|
| Discrete `argmax` may create gradient bias. | Experiment with straight‑through estimator vs. REINFORCE vs. soft Gumbel relaxation. |
| Stack overflow/underflow on rare controller errors. | Add pointer guards and overflow penalty to loss. |
| Pointer is non‑diff, limiting gradient flow for compositional tasks. | Compare with *soft pointer* variant where pointer is a learned distribution. |

---

#### 9 · Future Extensions  

* Add **QUEUE** and **MULT** tools for breadth‑first memory and multiplicative arithmetic.  
* Use **Program‑of‑Thought prompting** to let language models emit tool calls explicitly (*Neural Forth in a Transformer*).  
* Couple with **weight‑tying tricks** to share tool interface across modalities (vision tokens pushing features onto stack for multi‑step reasoning).

---

By cleanly **decoupling “what to do” (learned) from “how to do it” (frozen, deterministic)**, this line of work aims to push neural networks toward the **modular, compositional** regime long sought in neuro‑symbolic AI—while staying friendly to modern accelerator hardware and gradient‑based optimisation.
