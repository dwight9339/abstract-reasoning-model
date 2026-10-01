# Frozen Abstract Reasoning Core with Learnable Interfaces

## Research question

Can we train a reusable abstract reasoning system independently of the representations it reasons over?

More specifically:

> Can a new interface, trained only on simple examples from a new representation, learn to invoke a frozen reasoning core and thereby inherit reasoning capabilities that were never demonstrated during interface training?

A successful result would establish a meaningful separation between:

$$
\text{learning how to represent a problem}
$$

and

$$
\text{learning how to reason over that representation}.
$$

The initial goal is **not** to demonstrate a universal reasoning engine. It is to demonstrate reusable computation across independently learned interfaces.

---

## Core hypothesis

Train a reasoning model \(R\) entirely on abstract symbolic problems. Give it an evolving working state and allow the same learned computation to execute repeatedly:

$$
s_{t+1}=R(s_t,m_t)
$$

where \(s_t\) is working state and \(m_t\) is optional persistent/scratch memory.

The parameters of \(R\) remain fixed during inference, but execution length does not.

Easy problems may require:

$$
R^3
$$

while difficult problems may require:

$$
R^{30}.
$$

After training, freeze \(R\) permanently.

For each external representation—text, tables, diagrams, etc.—train a separate encoder:

$$
E_i(x_i)\rightarrow z
$$

which maps the external problem into the representation consumed by \(R\).

The complete system is:

$$
x_i
\rightarrow
E_i
\rightarrow
R^{\,T(x)}
\rightarrow
a.
$$

For the first experiment, keep output decoding trivial. The reasoning core should emit a constrained answer such as a Boolean value, entity ID, integer, or pointer. This minimizes the risk that a powerful decoder performs the reasoning itself.

---

## Experimental principle

The critical test is **capability inheritance**.

Suppose the frozen core has already learned a task family up to reasoning complexity \(d=30\).

Train a new adapter only using instances with:

$$
d\le4.
$$

Then evaluate the complete system on:

$$
d=5,10,20,30.
$$

If the adapter itself learned to solve the task, performance should deteriorate sharply outside its training regime.

If the adapter learned to translate problems into representations understood by the frozen reasoner, it should inherit some substantial portion of the core's existing capability.

That distinction is the central experiment.

---

## Phase 1: Build the abstract reasoning core

Start with a deliberately restricted set of tasks where the underlying structure is known exactly.

Recommended initial families are graph reachability, transitive relations, ordering constraints, shortest-path-style reasoning, simple constraint satisfaction, and rule-chain deduction.

Reasoning Gym is a useful substrate because it already provides more than 100 procedurally generated task environments, deterministic generation, adjustable task complexity, and algorithmic answer verification. Its dataset abstraction exposes the question, answer, metadata, and task-specific scoring function, so the underlying generators can be adapted without adopting its natural-language representation.

However, the first implementation should not simply tokenize Reasoning Gym's English questions.

Construct an intermediate canonical representation such as:

$$
\mathrm{EDGE}(X_1,X_2)
$$

$$
\mathrm{EDGE}(X_2,X_7)
$$

$$
\mathrm{QUERY\_REACHABLE}(X_1,X_7).
$$

Randomize entity IDs and, where possible, relation symbols between episodes to prevent persistent semantic associations.

Train the core exclusively on these abstract representations.

The first architecture should be intentionally simple: a small Transformer-like or recurrent-attention block whose parameters are reused across computation steps, plus an explicit scratch/working-state mechanism and either a learned halting mechanism or externally varied execution budget.

Train across a wide distribution of structural complexity so that the core itself demonstrates extrapolation across reasoning depth before proceeding.

---

## Phase 2: Establish a core capability curve

Before training any adapters, characterize what the core can actually do.

Measure accuracy as functions of problem size, required reasoning depth, memory demand, and execution budget.

The important output is a capability surface such as:

$$
P(\text{correct}\mid d,n,T)
$$

where \(d\) is logical depth, \(n\) is problem size, and \(T\) is allowed recurrent computation.

This distinguishes three failure modes:

insufficient learned procedure, insufficient execution time, and insufficient working memory.

The core should show that increasing \(T\) improves harder problems at least over some range. Otherwise the recurrent design is not buying useful adaptive computation.

Freeze the best checkpoint after this stage.

No later experiment may modify its weights.

---

## Phase 3: Generate equivalent multimodal presentations

For each canonical problem instance \(z\), generate several surface representations of the exact same underlying structure:

$$
z
\rightarrow
\begin{cases}
x_{\text{text}}\\
x_{\text{table}}\\
x_{\text{diagram}}
\end{cases}
$$

Text might express graph relationships as sentences.

Tables might express them as rows of source/relation/target entries.

Diagrams might render nodes and edges visually.

Because every presentation comes from the same canonical state, the experiment has exact knowledge of what each adapter ought to communicate to the reasoning system.

This paired construction is extremely valuable: representation differences can vary while underlying reasoning demand remains identical.

---

## Phase 4: Train constrained adapters

Train independent adapters:

$$
E_{\text{text}},
\quad
E_{\text{table}},
\quad
E_{\text{diagram}}.
$$

The reasoning core remains frozen.

Initially train adapters using only simple problems—for example \(d\le4\) and small graphs—even though the frozen core can solve substantially harder instances.

Adapters should be deliberately capacity constrained.

This is essential.

A sufficiently capable adapter could learn:

$$
x\rightarrow\text{solve}\rightarrow\text{encode answer}
$$

and render the reasoning core irrelevant.

Useful constraints include low parameter counts, shallow architectures, restricted output-token counts, no recurrent computation in the adapter, and a fixed-size bottleneck between adapter and core.

The exact restrictions should become ablations later, but Phase 1 should strongly bias toward making reasoning inside the adapter difficult.

---

## Phase 5: Primary evaluation — capability inheritance

Evaluate each adapter on increasingly difficult instances from the same structural families.

The strongest simple result would look like:

$$
\text{adapter training}: d\le4
$$

while:

$$
\text{successful inference}: d\gg4.
$$

Compare against at least four controls:

1. Adapter alone, with no reasoning core.
2. Same total parameter count trained end-to-end.
3. Frozen but randomly initialized core.
4. Core whose recurrent execution budget is restricted to adapter-training depths.

If the proposed system substantially exceeds all four, especially beyond the adapter training regime, that supports the claim that the adapter is accessing previously learned computation rather than reproducing it.

Also vary the reasoning budget at inference time.

If harder problems improve when the same frozen core is run for longer, that is evidence that the learned interface exposes the problem to an iterative procedure rather than merely retrieving a memorized output.

---

## Phase 6: Cross-representation equivalence

Present the same underlying instance through different adapters.

Compare not only final answers but internal core trajectories.

For example:

$$
E_{\text{text}}(x_t)
\rightarrow
s_0^{(t)}
$$

and:

$$
E_{\text{diagram}}(x_d)
\rightarrow
s_0^{(d)}.
$$

The vectors need not be numerically identical.

Instead test functional equivalence:

Do they cause similar downstream state evolution?

Can intermediate states produced through one interface be substituted for those produced through another?

Do probes recover the same entities and relations?

Does perturbing corresponding elements have equivalent effects?

Convergence toward representation-independent internal computation would considerably strengthen the result.

---

## Phase 7: New structural compositions

Passing the previous experiment demonstrates reuse across interfaces, but not yet broadly reusable reasoning.

Next, train the core on primitive reasoning families such as:

$$
A=\text{ordering},
\quad
B=\text{graph traversal},
\quad
C=\text{constraint propagation}.
$$

Then construct new domains whose solutions require novel compositions such as:

$$
A+B,
\quad
B+C,
\quad
A+B+C.
$$

The new adapter should receive only a relatively small number of examples.

The question becomes:

> Can the adapter learn how the unfamiliar domain maps onto computational primitives the core already possesses?

This is the point at which the experiment begins testing transfer of reasoning structure rather than merely transfer across presentation formats.

---

## Phase 8: Runtime-defined rules

The strongest extension is to stop keeping the world's rules fixed.

Each episode supplies:

$$
\text{entities}
+
\text{arbitrary relations}
+
\text{new transformation rules}
+
\text{initial state}
+
\text{query}.
$$

The semantics of symbols change every episode.

A successful core must therefore infer the operative structure from context and execute it rather than rely on persistent meanings.

This moves the project toward the broader hypothesis of a domain-independent reasoning machine.

---

## Training strategy

For the reasoning core, begin with supervised learning where canonical solutions or intermediate states can be generated automatically.

Then use verifiable-reward reinforcement learning for tasks where multiple reasoning trajectories may lead to the same solution.

Reasoning Gym already supports dynamically generated training samples and task-specific algorithmic scoring, and most of its datasets expose curriculum controls.

Do not require natural-language chain-of-thought.

The scientific target is the computation itself, not imitation of a verbal explanation.

For adapters, begin with paired supervision against canonical representations if needed to stabilize training, then transition toward task-level loss through the frozen core.

The final experiments should minimize direct supervision of the internal representation so the adapter is free to discover whatever interface works best.

---

## Key metrics

The primary metric is not ordinary test accuracy.

Define an **inheritance gap**:

$$
I(d)=
A_{\text{core+adapter}}(d)
-
A_{\text{adapter-only}}(d).
$$

Of particular interest is \(I(d)\) for reasoning depths substantially beyond adapter training.

Also track:

$$
\text{accuracy vs execution steps},
$$

$$
\text{accuracy vs adapter capacity},
$$

$$
\text{accuracy vs problem complexity},
$$

$$
\text{sample efficiency of new adapters},
$$

and:

$$
\text{transfer to unseen structural compositions}.
$$

A particularly compelling result would be a new adapter trained on hundreds or thousands—not millions—of simple examples that immediately accesses difficult capabilities already present in the frozen core.

---

## Failure criteria

The project should be considered unsuccessful in its initial claim if any of the following occur:

adapter-only baselines extrapolate just as well as core+adapter systems;

performance stops near the maximum complexity encountered during adapter training;

increasing frozen-core compute provides no benefit;

large adapter capacity is required before transfer appears;

random or minimally trained cores perform similarly;

or every new structural family requires substantial retraining of the core.

These outcomes would suggest that representation formation and reasoning are too entangled under the tested architecture to obtain the desired modularity.

That would still be scientifically informative.

---

## Implementation scope for the first prototype

Do not start with language models, multimodal foundation models, or hundreds of Reasoning Gym tasks.

Build one controlled structural domain first.

Recommended first target:

$$
\boxed{\text{directed graph reachability + relational composition}}
$$

because it offers exact representations, easy procedural generation, arbitrary scaling in difficulty, deterministic verification, natural text/table/diagram renderings, and clear separation between parsing and multi-step reasoning.

The first milestone is simply:

> A frozen recurrent symbolic core trained on graph reasoning receives problems through a text adapter trained only on small graphs and substantially outperforms adapter-only controls on much larger graphs.

Then reproduce the result using a table adapter.

Then a diagram adapter.

Only after all three succeed should the project expand to multiple reasoning families.

---

## Relationship to existing work

This project differs from ordinary reasoning fine-tuning because the reasoning system is trained independently of the eventual domain interface.

It differs from conventional adapters because the adapter is not intended to add reasoning capability; it is intended to expose a domain to capability already present elsewhere.

And it differs from model distillation because the primary experiment is architectural separation rather than compression.

Recent work such as Universal Reasoner provides evidence that reasoning-related computation can be implemented as a lightweight module attached to frozen language models and can show cross-domain transfer, which makes the broader modularity hypothesis less speculative. Our experiment asks a different and more controlled question: whether an independently trained interface can inherit the capability of a **frozen abstract recurrent reasoner**.

---

## Definition of initial success

The first paper-worthy result does not need to establish a universal reasoning core.

It needs to demonstrate:

$$
\boxed{
\text{A weakly trained new interface can unlock substantially stronger reasoning already contained in a frozen computational module.}
}
$$

Specifically, the new adapter should learn from simple instances, remain incapable of solving the hard problems independently, and nevertheless enable the frozen core to solve harder instances approaching the core's native capability range.

If that occurs across text, tables, and diagrams, it would provide evidence that **learning to understand a representation can be meaningfully separated from learning the computation performed over it**.

That would justify the much larger research program.
