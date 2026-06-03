# Artifact Subsystem Design in ARENA

## 1. Document Scope

This document defines the **implementation rationale, design responsibilities, and auditability requirements** of the artifact subsystem in ARENA.

Its purpose is not to restate the philosophy of AI-assisted analysis at a high level. That role belongs to a separate document (`ai-assisted-analysis.md`). The purpose here is to explain why the artifact subsystem must exist as a real control layer, what responsibilities it carries, and why its apparent weight is structurally necessary.

In ARENA, an artifact is not just an export output or a packaging convenience. The act of "sending something to AI" is itself treated as a design problem. If it is impossible to explain, after the fact:

- what data was sent,
- why that data was selected,
- in what order it was presented,
- under what structural assumptions it was framed,

then the resulting AI output is not fully auditable.

For that reason, the artifact subsystem is designed as both:

- an input delivery mechanism for AI-assisted analysis, and
- a control mechanism for later audit, comparison, re-validation, and failure isolation.

Its goal is to deliver **the right data, in a structure that minimizes misinterpretation, while preserving the ability to re-check every step**.

---

## 2. Purpose of the Artifact Subsystem

The artifact subsystem is not a simple export layer for handing outputs to generative AI.

Its purpose is to support AI-assisted analysis under conditions where models may fail because of:

- missing inputs,
- missing context,
- structural misreadings,
- priority confusion,
- model-specific hallucinations,
- overemphasis on dominant differences.

Because those failure modes are plausible, ARENA treats "AI handoff" as a controlled analytical stage rather than a casual transfer step.

The artifact subsystem therefore includes more than files. It includes the logic and metadata needed to preserve:

- what was selected,
- why it was selected,
- what should be read first,
- how the comparison structure should be understood,
- whether the same input can be reconstructed later,
- whether an observed failure came from input mismatch rather than model behavior.

This is why artifact design belongs to the core analytical path, not to a peripheral utility layer.

---

## 3. The Core Nature of the Subsystem

The artifact subsystem is not merely an output formatter.

Its essential role is to combine two functions:

1. **delivering data for AI analysis**, and
2. **making AI responses auditable, comparable, and re-validatable**.

To support those functions, the subsystem includes elements such as:

- file selection,
- priority control,
- structure-aware packaging,
- manifests,
- selection metadata,
- hashes,
- provenance / lineage,
- verify / replay,
- model-specific bundles.

Individually, these may look like small implementation details. Collectively, they form the design required to make AI-assisted analysis:

- reproducible,
- auditable,
- comparable,
- diagnosable (failure-isolable at the input level).

---

## 4. File Selection and Priority Control

Generative AI usage always operates under constraints. A model cannot reliably consume everything at once with equal quality. Context windows, input budget, and structural ambiguity all matter.

Therefore, the artifact subsystem must decide:

- which files are included,
- which files are most important,
- what should be read first,
- what should be treated as primary evidence,
- what should remain supplementary.

This is not just a matter of reducing volume. It is a matter of preserving analytical intent.

Without explicit selection and priority control, dominant differences tend to drive the model's interpretation, while smaller but meaningful differences may be buried. File selection and priority control therefore exist to express:

- what the comparison target really is,
- what the model should anchor on first,
- what counts as supporting versus primary material.

This makes the handoff more faithful to the human analyst's actual evaluation structure.

---

## 5. Structure-Aware Packaging

Sending the "right files" is not sufficient if their internal relationship remains ambiguous.

In multi-condition, multi-period, multi-metric analysis, the model may still misread the analytical frame unless the package itself encodes structural relationships explicitly. This includes:

- comparison boundaries,
- expected grouping,
- priority order,
- relationship between summary and supporting material,
- which outputs belong to the same analytical unit.

Structure-aware packaging exists to reduce misinterpretation without fully eliminating interpretive flexibility. The goal is not to force a single reading. The goal is to prevent avoidable structural confusion.

This is especially important when the analytical target includes small differences that can easily disappear behind more visually or statistically dominant signals.

---

## 6. Why Manifests and Selection Metadata Exist

Manifests and selection metadata are not incidental attachments.

They are **audit records for the AI input itself**.

Their function is to make it possible to explain, after the fact:

- which files were included,
- why they were included,
- in what priority order they were meant to be consumed,
- what analytical role they played inside the bundle.

This matters because AI output cannot be audited properly if the input conditions are opaque.

If a model makes a questionable claim, one of the first questions should be: *what exactly did it see, and under what intended structure?* Without manifests and selection metadata, that question becomes harder to answer. Re-validation also becomes weaker because the reconstructed bundle may differ from the original without anyone noticing.

In this sense, manifests and selection metadata are not mere documentation. They are part of the analytical control surface.

---

## 7. Hashes, Verify, and Replay

SHA256 hashes, verify mechanisms, and replay mechanisms exist so that the input condition itself can be treated as auditable and reproducible.

Their role is to answer questions such as:

- Was the bundle modified between creation and consumption?
- Is this really the same artifact that was used in a prior run?
- Can the same input be re-checked in another environment?
- Did the model behave differently, or did the input differ?
- Was a file missing, altered, or corrupted?

These mechanisms are not included for abstract strictness. They are included because AI output cannot be trusted or challenged properly unless the input can be re-established with confidence.

In ARENA, verification must start one step earlier than "was the answer good?" It must begin with "was the input condition the one we intended?" Verify / replay provides that basis.

---

## 8. Provenance and Lineage

Provenance / lineage exists so that artifact contents can be traced back through the upstream analytical path.

If an input file is ambiguous in origin, then when something goes wrong it becomes difficult to determine whether the problem came from:

- the selection decision,
- the packaging logic,
- the metric generation step,
- an upstream transformation,
- the model's own interpretation behavior.

Lineage therefore preserves the unit of investigation.

It allows a file inside an artifact to remain connected to:

- the process that produced it,
- the stage it belongs to,
- the upstream source from which it was derived.

Combined with manifests and hashes, provenance / lineage turns the artifact from a simple delivery package into a traceable validation unit.

---

## 9. Model-Specific Bundles

Model-specific bundle separation for systems such as `GPT`, `Claude`, `Gemini`, or `Grok` is not only a convenience choice.

It is a design decision made for **cross-checking**.

Different models differ in context retention tendencies, summarization style, omission patterns, hallucination behavior, and preference for dominant versus subtle patterns. These differences are useful when explicitly compared.

To make that comparison meaningful, the system must preserve comparable input units across models. Model-specific bundles provide that separation. Their role is not merely to maintain compatibility. Their role is to preserve the ability to ask:

- what did each model see,
- where did they agree,
- where did they diverge,
- which claims appeared in only one model's output,
- what should be re-validated next.

The key point is that the value of cross-model usage lies not only in overlap, but in disagreement that can be turned into a next-step validation task.

---

## 10. Operational Limits of Cross-Checking

Cross-checking is powerful, but it introduces cost. This matters for artifact design because **the subsystem must support multiple validation styles rather than assume a single fixed workflow**.

Once a disagreement is found, a valid follow-up often requires:

- reconstructing or refining the input,
- deciding what the disagreement actually means,
- generating a targeted re-validation prompt or bundle,
- comparing the second-round outputs,
- tracing the disputed claim back to source data.

This process takes time and attention.

Sometimes it is more efficient for a human to lightly screen a single-model answer first and discard obviously unsupported claims before entering full cross-model comparison. That can improve reliability, but it also increases review cost.

For that reason, the artifact subsystem is designed to preserve the conditions necessary for disciplined follow-up across different validation patterns — full cross-model comparison, single-model screening followed by targeted cross-checking, or iterative refinement of a single bundle. Its job is not to pretend that all follow-up is cheap.

---

## 11. Connection to the Bottleneck Principle

The weight of the artifact subsystem is not an accident and not a design mistake.

ARENA assumes that the quality of the full analysis is capped by its weakest stage. That principle applies to AI-assisted interpretation as much as it applies to data collection or metric generation.

Even if upstream stages are rigorous, the final interpretation becomes constrained if the AI input stage fails to preserve:

- structure,
- priority,
- intended comparison boundaries,
- evidential traceability.

Therefore, the artifact subsystem should not be treated as a post-processing convenience layer. It is the layer that extends pipeline consistency into the interpretation stage.

Its complexity is the consequence of trying to preserve that consistency under real model limitations.

---

## 12. Current Implementation Boundary

The artifact subsystem is the main control layer for current public AI-assisted analysis.

It does not exist in isolation. ARENA also includes a database-oriented layer intended to function as a knowledge base for baseline sharing, discovery of new analytical viewpoints, and extraction of higher-plausibility inference candidates. The purpose, current status, and design rationale of this database layer are described in `ai-assisted-analysis.md` §11.

In the context of artifact design, the relevant point is that the artifact subsystem sits **upstream** of this database layer. It generates input units that are structured, auditable, and reproducible enough to be trusted as reusable analytical records later.

Since `v0.3.1`, the synthesis database layer is part of the public release layer. As of `v0.4.0`, the Stage 9 evidence synthesis layer also produces reviewable evidence artifacts that can feed later synthesis and AI-assisted workflows. Both remain downstream of artifact generation and are still evolving. Therefore, the practical focus of this document remains the artifact-centered control model that feeds those layers safely:

- input control,
- auditability,
- reproducibility,
- replayability,
- cross-model comparability.

---

## 13. Audit Scope of This Document

The audit target of this document is whether the artifact subsystem provides sufficient **implementation rationale and design responsibility coverage** to make AI-assisted analysis operationally credible.

The main question is not "are many features listed?" The main question is whether the subsystem is defined coherently as a control layer that supports:

- fixed and explainable input conditions,
- traceable selection decisions,
- reproducible bundles,
- re-validation capability,
- failure isolation,
- comparable cross-model usage.

This means the audit should examine whether each major component is justified in terms of failure prevention and analytical control:

- why selection exists,
- why priority exists,
- why structure-aware packaging exists,
- why manifest and metadata exist,
- why hashes and verify / replay exist,
- why lineage exists,
- why model-specific bundles exist.

If those elements are present only as implementation fragments but their necessity is not explained, then the subsystem will appear heavy without appearing justified. This document is therefore audited as the explanation of **how AI-assisted analysis is made auditable and reproducible in practice**.

---

## 14. Conclusion

The artifact subsystem exists to combine two requirements that must coexist:

- data must be delivered to AI in a usable analytical form, and
- the resulting interpretation process must remain auditable, comparable, and re-validatable.

That is why the subsystem includes manifests, selection metadata, hashes, provenance / lineage, verify / replay, and model-specific bundle handling. These are not ornamental details. Together, they form the implementation basis that makes AI-assisted analysis in ARENA:

- reproducible,
- auditable,
- traceable,
- diagnosable.

The artifact subsystem is therefore not a peripheral export feature. It is one of the core layers that make the ARENA analysis workflow defensible.
