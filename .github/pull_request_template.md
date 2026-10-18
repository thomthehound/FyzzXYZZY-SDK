## Owner and source contract

<!-- Name the actual production caller and state owner. Describe the source method or equivalent SDK responsibility. If no production code changes, identify the exact test/build contract. -->

## Runtime cost and scheduling

<!-- Identify GPU submissions, alloc/growth, staging, waits, pipeline and descriptor lifetimes, and any new per-frame work. For non-runtime changes explain why these are unchanged. -->

## Integration evidence

<!-- What tests prove the changed behavior, on what backends? Differentiate unit/helper tests from actual scene/engine execution and disclose missing acceptance. -->

## Redundancy and retirement

<!-- Name overlapping implementations/state and what this change reuses, removes or migrates. If none, explain how this was checked. -->

## Source fidelity and deviations

<!-- Use exactly one: source-translation, semantics-preserving-lowering, non-production-only, not-applicable.
For native scene/solver/contact/broadphase production changes, only the first two are admissible.
Identify the original PhysX method and file; state every semantic/scheduling/numerical deviation
and the specific parity evidence. Do not pass a provisional solver as source-equivalent. -->
Fidelity classification:

PhysX source:

Departures:

Parity evidence:
