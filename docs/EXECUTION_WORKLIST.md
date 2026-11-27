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
- [x] Separate recorded and submitted resource lifetime, provisional PCM
  history relocation, nonblocking journal repair, and readback cancellation.
  Join rejected source tokens at one completion boundary and preserve
  immutable mapped inputs across replacement.
- [x] Retain descriptor and endpoint state across packed singleton PGS
  iterations; publish contact body deltas at descriptor boundaries.
  Keep the biased motion snapshot distinct from final velocity state.
- [x] Synchronize serial singleton partition runs only at cooperative
  motion capture and integration; retain dependencies between concurrent
  conflict partitions.
- [x] Retain single-row packed joint Jacobians, phase coefficients and
  accumulated impulse across the solve. Use the same row operation as
  streamed multi-row joints and publish the final impulse once.
- [x] Retain single-patch contact metadata across packed iterations and
  publish final friction anchors directly from solved tangent rows.
  Stream variable-size manifolds without per-lane row arrays.
- [x] Pack normal and tangent numerical rows to 56 and 52 bytes.
  Resolve the shared contact impulse limit from its preparation record
  once per descriptor span rather than duplicating it per normal row.
- [x] Stream serial singleton colors as contiguous descriptor runs.
  Preserve ordering and synchronize only at run boundaries, cooperative
  motion capture, and integration. All-singleton tasks require no
  per-iteration color-boundary loads.
- [x] Retain responding endpoints across consecutive serial descriptors
  by solver-body ordinal. Publish evicted endpoints and flush retained
  state at the run boundary; preserve per-descriptor inverse masses.
- [x] Execute each phase of single-color fused tasks under exclusive
  descriptor ownership. Retain numerical state within the phase and
  synchronize only for cooperative motion capture and integration.
- [x] Parallelize uncontended static endpoint reservations in large
  color tasks. Preserve encounter rank for repeated endpoints and emit
  single-color descriptors directly without histogram or atomic scatter.
- [x] Retain serial-run endpoint state across iteration boundaries in
  fully serial fused tasks. Flush at motion capture and integration
  rather than at each iteration; retain tiled run boundaries.
- [x] Relocate device-admitted contact windows during descriptor
  expansion using Sc's exact prefix. Share relocation between colored
  and packed tasks without capacity staging or a separate traversal.
- [x] Emit retained-color descriptors during expansion and preserve
  unchanged raw records. Keep packed descriptors local through
  preparation, solve, and integration.
- [x] Omit sorted descriptor, partition, raw descriptor, cursor and
  endpoint coloring storage for host-admitted packed-only frames.
  Keep source tokens and the authored descriptor table; mixed and
  device-admitted frames retain their required coloring capacity.
- [x] Place device-only coloring workspace beyond the input packet.
  Upload source tokens, authored descriptors and schedule inputs in
  one contiguous copy, without staging GPU-owned scratch.
- [x] Initialize only admitted color histogram entries. Include sorted
  descriptor and coloring workspace locations in retained schedule
  identity; validate the compact result extent independently of uploads.
- [x] Reuse partition slots for static reservation lists, histogram
  counts and scatter offsets. Atomic scatter leaves exact partition
  ends in place; reserve separate wide-color lists only for resident
  tiled scheduling.
- [x] Bound resident wide-color lists by full 128-descriptor groups
  and derive list offsets from disjoint task descriptor ranges.
  Initialize endpoint priorities only at their consuming phases.
- [x] Preserve unchanged sorted descriptors and retained task mode and
  wide-color lists. Rebuild global tiles from locally retained or
  newly colored tasks without rescanning unchanged partitions.
- [x] Bound the shared resident partition tile pool by wide-color
  widths and intervening narrow runs, rather than response count.
- [x] Declare endpoint remapping and tiled numerical preparation
  accesses by their actual arena tables and source bindings, including
  friction history, contact extents and task error publication.
- [x] Declare coloring and tile construction access ranges separately
  from numerical solver rows. Preserve combined body initialization,
  resident schedule reuse, task errors and indirect argument writes.
- [x] Compose fused and packed dependencies from their numerical phases.
  Declare integration result streams and external shape and actor state
  explicitly; free flight excludes constraint rows and solver inertia.
- [x] Share Vulkan descriptor sets across dispatches with identical
  physical bindings. Preserve dispatch-local access tracking and push
  constants, and skip redundant native descriptor binds.
- [x] Normalize Vulkan operation aliases into exact disjoint access
  intervals before updating queue hazard history. Coalesce only
  adjacent intervals with identical scope and write intent.
- [x] Use Vulkan Synchronization2 with individual barrier stage and
  access scopes, exact copy and storage access masks, and 64-bit
  dependency history across submissions.
- [x] Use D3D12 enhanced buffer barriers and access epochs for compute,
  copy and indirect dispatch dependencies. Copy non-overlapping buffer
  regions directly, without relocation staging.

- [x] Reuse validated dispatch access ranges within each native batch.
  Normalize Vulkan aliases and group D3D12 buffer ranges once per
  matching descriptor/access declaration; retain per-operation hazards.

- [x] Resolve and validate physical views once per distinct native
  descriptor table. Retain dispatch-local shape and access validation;
  key D3D12 heap reuse by the actual unique table layout.

- [x] Resolve dependency ranges to dense batch resource ordinals.
  Avoid unchanged Vulkan interval mutations and D3D12 reader-range
  maintenance after an epoch admits its first writer.

- [x] Share frame-local Dy phase resource declarations between numerical
  launches. Keep per-launch parameters and indirect arguments independent,
  and retain source-root lifetime checks for shared declarations.

- [x] Retain prepared table and range identities for shared declarations.
  Alternating numerical phases and indirect chunks reuse their distinct
  plans without repeating physical table matching or range normalization.

- [x] Store Dy launch parameters in a contiguous frame-local word store.
  Launches retain immutable slices through recording, upload expansion
  and submission; validate their extents independently of access-plan reuse.

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
