<!-- SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Execution architecture worklist

The target is a complete, independent CPU/Vulkan/D3D12 physics engine.
Execution follows numerical dependencies, not module boundaries. CPU
orchestration retains authored identities, filtering, callbacks and public
publication; dense numerical work remains on the device where beneficial.
Optimize end-to-end frame latency, throughput, memory traffic and numerical
quality rather than individual dispatch or barrier counts.

## Completed foundations

- [x] Establish ordered source execution, device numerical admission,
  resident actor/contact state and independent witness/result publication.
- [x] Reduce redundant numerical preparation, actor mirror traversal and
  friction-anchor packing.
- [x] Pack numerical inputs and source upload snapshots into native transfer
  storage; publish ordinal readback directly into its consumer.
- [x] Reuse readback, upload and frame-input storage according to recent
  demand; preserve live history when contracting numerical arenas.
- [x] Combine compatible transfer regions and retain unchanged native
  pipeline/descriptor bindings.
- [x] Track Vulkan byte-range hazards and producer visibility; detect
  conflicting D3D12 UAV reader/writer ranges.
- [x] Use one Vulkan device command recorder for copies and dispatches.
- [x] Use D3D12 buffer promotion and decay at submission boundaries,
  retaining intra-submission transition and UAV dependencies.
- [x] Skip unused LFS payloads during CI configuration.

These are implementation milestones, not hardware-performance acceptance.

## Execution and lifetime work

- [ ] Audit the frame critical path using existing profiling: producer,
  consumer, data residency, observation points, dispatch boundaries,
  intermediate materialization, and rollback/lifetime constraints.
  Preserve genuine overlap between CPU preparation and GPU work.
- [ ] Separate physical allocation, shader descriptor and declared access
  extents. Exclude alignment padding and identify phase-specific reads and
  writes within numerical arenas. Validate bounds before recording work;
  use no additional descriptors or copies solely to communicate hazards.
- [ ] Extend Vulkan dependency ownership across submissions. Eliminate
  blanket dependencies where exact producer/consumer ordering suffices.
  Preserve host visibility, fence retirement, deferred frees, resource
  reuse and failed-submission behavior.
- [ ] Audit transient arena peaks, initialization, scratch reuse and
  placement. Keep persistent histories separate in lifetime from frame
  scratch where useful; avoid unnecessary CPU mirrors, staging, copies,
  and host/device transfers. Keep hot numerical data local.
- [ ] Audit shader work and binding granularity. Keep pair- or group-local
  intermediates in registers/shared memory where practical; reduce
  redundant dispatch, compaction, scan and publication work.
- [ ] Exploit persistent accurate connectivity and admission. Reuse
  component membership and ordering when responding topology is unchanged.
  Handle additions, edge removals, splits, activation and retirement
  exactly. Rebuild affected components without a duplicate graph owner.
- [ ] Improve wide-island execution: determine real per-body constraint
  dependencies, partition and iteration ordering, and cross-cluster
  communication. Preserve PGS semantics, position/velocity phases,
  convergence, friction, and stable publication. Do not rely on
  cross-workgroup spin-wait forward progress.
- [ ] Revisit broadphase-to-PCM overlap. Continue admitted persistent
  numerical work independently of host reconciliation where valid;
  preserve filtering, new/retired manager identity and touch semantics.
  Evaluate sparse CPU versus resident GPU work by total frame cost.
- [ ] Evaluate pair-local PCM/GJK/EPA fusion, worklist necessity and
  cooperative support searches for complex convex geometry. Software
  adapter validation is not a hardware scheduling baseline.
- [ ] Review rollback, retirement, allocation lifetime and publication
  across all changed paths. Remove obsolete branches and duplicated
  ownership; retain only production execution paths with a defined role.
- [ ] Tighten regression coverage around observable physics, dependency
  ordering, memory lifetime, mixed workloads and rejected transactions.
  Remove redundant low-signal tests; keep independent numerical checks.
- [ ] Review CI caching, repeated builds and tests, shader compilation,
  suite partitioning and runner reliability without reducing coverage.
- [ ] Complete final architectural and capability review before physical
  hardware qualification. Record remaining feature gaps separately from
  execution infrastructure.

## Qualification

Measure physical Vulkan and D3D12 GPUs across sparse/free bodies, compact
independent islands, large connected and mixed islands, persistent versus
changing contacts, and convex workloads. Record complete frame latency,
CPU preparation, device dispatch timing, queue occupancy, synchronization,
transfer and memory working set, solver convergence and numerical outcomes.
Use the same scene and configuration for comparisons. Software execution
qualifies API and numerical behavior, not physical GPU performance.

No optimization is accepted on dispatch count alone. A stage boundary
requires a real numerical dependency, device-wide coordination, host
observation or resource lifetime constraint. Maintain one authoritative
scene and numerical state; do not add a general-purpose graph runtime or
a shadow solver.
