# AI-Assisted Analysis in ARENA

## 1. Document Scope

This document defines the **principles, decision criteria, and operating model** for how ARENA uses generative AI.

Its purpose is not to describe implementation details or to enumerate subsystem features. Instead, it explains the design position behind AI usage in ARENA: what role AI is allowed to play, what it is not trusted to do on its own, and which constraints must remain visible throughout the analysis process.

In ARENA, generative AI is **not** treated as a final-answer engine or as an automatic source of truth. It is treated as a support resource for hypothesis generation, perspective expansion, difference detection, and candidate discovery for further validation. Final judgment remains with the human operator.

The value of AI in this context is not decision outsourcing. The value is **reducing interpretation workload while expanding the set of candidate viewpoints worth checking**.

This document intentionally does **not** cover the concrete mechanics of the artifact subsystem. Components such as manifests, selection metadata, hashes, verify / replay, and model-specific bundles are covered in a separate document (`artifact-design.md`). What matters here is the higher-level rationale that makes those mechanisms necessary in the first place.

---

## 2. Design Position on AI Usage

ARENA does not use generative AI merely as an auto-summarizer. It uses AI as an analytical collaborator under explicit constraints.

The design goal is:

- **compress human effort**,
- **widen the search space** of possible interpretations,
- **preserve human control** over acceptance and rejection.

The central assumption behind this goal is simple: AI can sometimes surface viewpoints, candidate interpretations, and comparison axes that a human would not enumerate as quickly, especially when many outputs, metrics, periods, and parameter variants must be compared continuously. In a solo or small-team workflow, the human cost of repeatedly reading large volumes of statistical output is high. AI can reduce that cost by proposing candidate interpretations earlier in the process.

This does **not** mean the system trusts AI as a final evaluator. It means AI is useful as a support resource for:

- hypothesis generation,
- initial interpretation scaffolding,
- difference detection,
- candidate selection for follow-up checks.

At least within this project's scope, generative AI can propose candidate interpretations across multiple columns and metrics faster than exhaustive manual review. That speed, combined with explicit human control over what gets accepted, is where the value lies.

---

## 3. How This Differs from Typical AI Usage

The following differences are not criticisms of standard AI usage patterns. They reflect the fact that ARENA's analytical targets demand different guarantees.

In many common AI workflows, the operating assumptions are:

- a single prompt is often enough,
- speed is prioritized over verification,
- approximate correctness is often sufficient,
- agreement is treated as sufficient confidence,
- re-checking is limited.

ARENA operates differently:

- AI outputs are treated as hypotheses, not conclusions,
- agreement is treated as a baseline, not as proof,
- disagreement is explicitly preserved,
- small differences may still matter,
- re-validation loops are assumed from the start.

The key definition is:

> AI output is a hypothesis, not a conclusion.

This definition changes how outputs are used. AI responses are not handled as pass/fail judgments. They are decomposed into:

- baseline claims,
- disputed claims,
- novel claims,
- follow-up validation targets.

---

## 4. Why Raw Bulk Input Was Not Enough

This design position was not chosen abstractly. It emerged from actual use.

At an earlier stage, processed statistical outputs and generated analysis files were bundled and handed to generative AI more or less as-is. This worked to a degree, but the model did not reliably surface the specific differences the human analyst intended to evaluate. The quality of interpretation was unstable.

A dominant example was a large hardware gap such as `rtl-sdr -> airspy`. When that kind of large difference exists, the model naturally tends to treat it as the primary story. That is reasonable at a coarse level, but it becomes a problem when the analytical target is not the dominant gap itself, but smaller meaningful differences such as:

- a ~10% gain from a parameter adjustment,
- a ~5% difference caused by a short RF-path component change,
- a subtle but repeatable shift in a constrained comparison window.

The problem is not that AI is "wrong" in a generic sense. The problem is that **if processed outputs are passed without explicit structural control, the comparison structure the human cares about may not survive the handoff**. Without priority and structure built into the input, dominant differences naturally absorb the model's attention. Smaller but meaningful differences may be flattened, deprioritized, or omitted.

---

## 5. Why Structured Tables Are Preferred over Images

ARENA prefers CSV and other structured tabular inputs whenever possible.

The reason is not aesthetic. The reason is stability of interpretation.

Comparisons involving multiple metrics, multiple conditions, period differences, parameter variants, and controlled deltas are generally easier to communicate through structured data than through images alone.

Charts and PNG outputs are useful as supplementary materials, but they are less reliable as a primary input medium when the task depends on preserving comparison axes precisely. A model may read the same image in multiple plausible ways, especially when visually dominant patterns overshadow smaller differences.

For that reason, ARENA treats images as supporting context, not as the default input layer. The primary input should preserve:

- explicit variables,
- comparison targets,
- ordering,
- condition boundaries,
- interpretable relationships.

---

## 6. Why ARENA Does Not Center the Workflow on Full API Automation

ARENA does not center its AI workflow on fully automated API-first execution. The core reason is that **iterative, exploratory interaction with controlled inputs produces better analytical results than a single closed pipeline**.

There are several specific reasons.

First, repeated validation, re-prompting, cross-model comparison, and follow-up questioning can become expensive in an automation-heavy design.

Second, the desired value is not only template completion or one-shot summarization. ARENA expects AI to produce fresh angles, alternative comparison frames, candidate missing data, and hypotheses worth testing. That kind of output benefits from iterative control.

Third, the problem is not whether data can be sent to AI. The problem is whether the **structure, priority, and context of that data are controlled before it is sent**. Without that control, dominant differences tend to absorb the model's narrative while smaller but analytically meaningful signals are buried.

Long-context accumulation introduces another problem: useful history can help, but stale assumptions, prior interpretations, and irrelevant context can also distort the next answer.

For these reasons, ARENA prioritizes:

- controlled input construction,
- exploratory interaction,
- explicit re-validation,
- human-managed interpretation loops,

over full automation as a default.

---

## 7. Core Workflow

AI-assisted analysis in ARENA is an iterative process, not a one-pass procedure.

A typical loop is:

1. Generate multiple candidate interpretations.
2. Extract the overlapping parts.
3. Detect differences and disagreements.
4. Identify what additional data would clarify the disagreement.
5. Run another analysis round.
6. Compare the result against the original data.

The purpose of using multiple outputs is not simply "to ask more than once." Each cycle converts raw AI output into structured analytical work:

- overlap becomes baseline,
- disagreement becomes a validation target,
- missing support becomes a data requirement,
- re-analysis becomes explicit follow-up work.

---

## 8. Cross-Model Validation

Cross-model usage in ARENA is **not** majority voting.

The operating rule is:

- agreement → baseline,
- disagreement → validation target.

The value of cross-checking lies less in agreement itself and more in the ability to identify:

- hallucinations,
- unsupported claims,
- model-specific distortions,
- potentially novel analytical viewpoints,
- missing data requirements.

Different models exhibit different tendencies in context retention, summarization style, omission patterns, overconfident interpretation, and hallucination behavior. That difference is useful.

If two models receive the same or near-identical inputs, their convergence and divergence become analytical signals. The most important case is not "both said the same thing." The most important case is "one surfaced a claim the other did not," because that creates a concrete re-validation task.

This does not make cross-checking free. Once a disagreement is found, someone still has to reconstruct the follow-up input, decide what the disagreement actually means, run the next validation round, compare the new outputs, and trace the claim back to source data.

That loop has real cost. In some cases, it is more efficient for a human to quickly discard obviously unsupported single-model outputs before full cross-model analysis begins. That can improve reliability, but it also increases review burden. The design therefore treats cross-checking as powerful but costly.

---

## 9. Bottleneck Principle and End-to-End Consistency

ARENA treats the following as a design constraint:

> The quality of the overall analysis is capped by its weakest stage.

This principle applies even when upstream stages are strong.

Data collection may be precise. Metric generation may be sound. Statistical evaluation may be rigorous. But if the AI input stage collapses the structure, distorts the priorities, or obscures the comparison logic, then the final interpretation is constrained by that weakness.

This has a direct consequence:

- AI input control is not optional,
- artifact construction is not a cosmetic layer,
- pipeline consistency must extend through the interpretation stage.

The artifact subsystem is therefore necessary not because the system was over-engineered, but because analytical consistency must survive the handoff into AI-assisted interpretation.

---

## 10. Why This Is Not a Common Pattern

This approach is not common for understandable reasons.

It is expensive in time. It increases cognitive overhead. In many tasks, small differences do not matter enough to justify this level of control. In many business or productivity use cases, 80% correctness is good enough and explicit re-validation is not worth the cost.

ARENA is an exception because:

- small differences can matter,
- reproducibility matters,
- interpretation itself is part of the evaluation target,
- disagreement is analytically useful rather than merely inconvenient.

That is why the workflow is heavier than a conventional AI usage pattern.

---

## 11. Current Implementation Scope, Including the Database Layer

ARENA also includes a database-oriented layer, but its purpose is often misunderstood.

The database is **not** intended merely as storage for past outputs. It is intended as a **knowledge base for baseline sharing, discovery of new analytical angles, and extraction of higher-plausibility inference candidates**.

Its role is to structure and preserve:

- past claims,
- supporting grounds,
- detected differences,
- limitations,
- unresolved issues,
- additional data requirements,
- reliability scores or coarse categories (such as `low`, `mid`, `high`) attached to analytical statements.

This makes it possible to do more than look up old results. It allows later analysis to compare against shared baselines, identify what is genuinely new, and search for underexplored viewpoints more efficiently.

The longer-term goal is to feed this structured layer back into AI-assisted workflows so that the model can help prioritize claims with stronger support, unresolved items worth re-checking, and viewpoints with higher potential analytical value. The purpose is not to replace human judgment. The purpose is to reduce human review burden while accelerating movement toward better-grounded inference.

As of `v0.3.1`, this database layer is included in the public release layer through the synthesis subsystem. It is still an evolving review-oriented layer rather than a public automation endpoint. Therefore, current public-facing AI-assisted analysis in ARENA should still be understood as centered on:

- artifact-mediated input control,
- explicit re-validation,
- human final judgment,

with the database layer positioned as an emerging knowledge substrate rather than a final public automation dependency.

---

## 12. Audit Scope of This Document

The audit target of this document is whether the **principles, decision criteria, and operating rules** for AI usage in ARENA are defined consistently.

What should be checked is not whether every implementation feature is listed, but whether the overall position is coherent:

- AI is treated as a hypothesis generator rather than a final arbiter,
- disagreement is preserved rather than erased,
- cross-model use is framed as validation rather than majority voting,
- small differences remain analytically visible,
- human final judgment remains explicit,
- the bottleneck principle is consistently connected to the need for input control.

In other words, this document should be audited as a definition of what ARENA trusts, what it refuses to trust automatically, what must be re-validated, and where responsibility remains human.

If that design position is vague, then the artifact subsystem will appear heavier than necessary and its rationale will be obscured. This document therefore functions as the conceptual precondition for the implementation design.

---

## 13. Conclusion

In ARENA, generative AI is not a truth engine. It is a hypothesis generator.

Those hypotheses are evaluated through:

- difference detection,
- re-validation loops,
- comparison against original data,
- human judgment.

The most important requirement is not convenience. It is preservation of structure and consistency across the full path from pipeline output to interpretation.

AI-assisted analysis in ARENA therefore means something specific: **integrating re-validatable hypothesis generation into the analysis workflow without surrendering analytical responsibility**.
