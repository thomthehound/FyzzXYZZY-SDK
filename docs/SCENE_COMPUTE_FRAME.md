# Scene-owned compute frame

## Execution authority

The native Scene owns one simulation transaction. CPU code owns persistent
actor/shape identity, authored changes, filtering, stable contact-manager
handles, application callbacks, and commit/rollback. The CPU and GPU execute
stages according to input residency, parallelism, and end-to-end cost.
The device does not own a second scene,
contact-manager registry, island lifetime, or command scheduler.

The Scene builds an ordered dependency graph. The compute backend records
that graph into native Vulkan or D3D12 command lists, with barriers and
queue fences owned by the backend. A function call, mapped pointer, or
shader dispatch is not a completion fence.

The frame must remain numerically correct with a CPU dynamics backend.
CPU execution is an independent correctness reference and the execution
path for unsupported GPU devices; it is not a shadow solver run alongside
GPU dynamics.

## Frame phases

| Phase | CPU authority | Device work | Host observation |
| --- | --- | --- | --- |
| Admit | Validate authored edits and sparse identities | Consume compact input changes | None |
| Broadphase | Own actor/shape/filter identities and explicit CPU SAP when selected | GPU SAP tracks bounds and overlap history when selected | Current GPU SAP fetches report deltas before Sc admission; target: only topology changes require Sc reconciliation |
| Contact | Resolve new managers, mixed Gu contacts, callbacks and CPU correlation consumers | PCM witnesses, touch history, numerical contact stream | Current: CPU correlation joins GPU witnesses; target: device consumers use resident records directly |
| Island and constraint preparation | Own stable graph and admit required task/identity changes | Build transient numerical components, partitions and rows where useful | CPU-prepared tasks or device admission, not both without a consumer |
| Dynamics | Own timestep, requested outputs and commit policy | PGS, integration, sleep and force/impulse reductions | Final publication |
| Publish | Validate the completed transaction; update required public CPU views | Write status, deltas and requested results | One completion join on the uninterrupted path; earlier joins only for actual CPU dependencies |

A GPU-generated topology delta cannot become an unadmitted contact manager.
Sc currently fetches SAP reports and joins GPU contact witnesses when CPU
correlation needs them. The production Scene has not yet removed that general
PCM-to-Sc join. Stable admitted identities permit device-dependent numerical
work to continue without a host stop once the required source inputs are
ready. A device gate must reject or suppress work needing new identities
until Sc reconciles the delta in the same transaction. Contact touch, not
speculative broadphase overlap, determines accurate connectivity.

Host callbacks, mixed CPU Gu interactions, externally requested body state,
and failure recovery may require additional explicit boundaries. Their
consumers and synchronization edges are part of the frame contract, not
unconditional waits added by individual kernels.

## Work placement and synchronization

CPU SAP is a candidate for sparse endpoint crossings when bounds are already
available to the CPU; GPU SAP is a candidate for dynamic, resident workloads.
A standalone SAP benchmark is insufficient: account for downloading bounds,
re-uploading pairs, Sc reconciliation and the location of downstream PCM.
The current native GPU World selects GPU SAP at construction; CPU SAP feeding
GPU PCM/Dy is not yet an equivalent production configuration and must not be
represented as measured. Avoid automatic placement changes without physical
crossover evidence and a proven single-owner history transition.

For each stage, establish input and output residency and the first real CPU
consumer before adding a synchronization edge. Queue dependency order is
sufficient between GPU producers and GPU consumers. Fence only for CPU
observation, resource reuse, rollback or public publication. CPU work may
prepare independent authored records during an outstanding GPU submission.
Avoid dispatches solely for orchestration or copying records that could be
consumed from an existing device allocation.

## Memory and access

Allocation policy is based on **access pattern and physical placement**,
not a blanket requirement that all GPU data be host mapped.

- **CPU-authored frame input:** compact, sequentially written control and
  geometry records. On compatible UMA resources, write directly into a
  GPU-local, CPU-mappable arena, unmap before GPU consumption, and bind
  that same allocation. On other devices, explicitly stage into GPU-local
  memory without maintaining a duplicate persistent CPU/GPU registry.
- **Device numerical state:** broadphase scratch, PCM witnesses, contact
  history, island worklists, constraint rows, and solver motion remain in
  GPU-preferred storage. Dependent passes exchange device addresses or
  offsets; they do not materialize host mirrors between passes.
- **Public output:** small completion/delta records can occupy CPU-readable
  mapped allocations. Bulk pose, impulse and contact readbacks are selected
  by real public consumers, not by the existence of a CPU scene.
- **Stable allocations:** capacities grow as needed. Ordinary frames reuse
  resources and descriptor bindings. Retire an old resource after all
  submitted GPU readers pass its queue fence.

Physical UMA avoids a discrete transfer only when both processors access
the same eligible allocation. It does not waive host/device visibility,
cache maintenance, barriers, or fence ordering. Vulkan direct frame inputs
require a GPU-local HOST_VISIBLE|HOST_COHERENT buffer type on an integrated
adapter; D3D12 requires a probed GPU-writable CPU-mappable custom UMA
resource. Device-local hot arrays remain the preferred numerical path.

## Transaction and continuation

The Scene assigns one epoch to authored input, device work, and public
publication. The GPU writes transient results into epoch-owned work
resources. Persistent histories use a single owner and an explicit
commit/undo boundary. A failed epoch cannot expose speculative contacts,
motion, or events through the public Scene.

The current production path joins SAP found/lost reports for Sc admission
and GPU PCM witnesses for CPU correlation. The Scene explicitly selects
compact device admission via set_gpu_dy_device_admission_enabled(): Sc passes
stable contact and joint identities, the GPU reads resident physical touch,
and Dy consumes generated task/body ranges without host task/ordinal uploads.
When the Scene selects device admission, the island work and Dy solve enter
one ordered GPU submission without an intermediate host wait. Island graphs
with at most 32 nodes and 32 edges build their task records in one dispatch;
larger graphs use parallel union, prefix and response ordering.
Wide islands use their separate device-partition execution path; the compact
admission selection rejects them. Sc otherwise reuses CPU-prepared
accurate-island task boundaries. Independent one-constraint islands with one
or two responding bodies execute in packed GPU lanes; dependent compact
islands use fused workgroups. Device-generated roots are numerical task IDs,
not replacements for Sc's persistent island identities. The Scene still
prepares solver inputs, joins GPU PCM witnesses for CPU friction correlation,
and reads back Dy scheduling and numerical results for publication.
A continuous PCM-to-Dy numerical path without a host witness dependency
remains open; compact admission alone does not remove that synchronization.

When topology requires CPU reconciliation, the device produces a compact
reason and delta stream, and suppresses dependent work. Sc validates and
admits the delta before recording the remaining commands. Restarting or
continuing this transaction must not duplicate already applied persistent
history updates. All resource references remain alive until the relevant
GPU fence has completed.

## Delivery gates

A scheduling or memory-placement change is accepted only with:

1. Exact source-owner and rollback behavior for same-frame found/lost,
   sleep/wake, mixed Gu, joint-only and large connected components.
2. No second contact registry, shadow solver, independently advancing
   stream scheduler, or unconditional bulk device-to-host witness transfer.
3. Hardware telemetry for whole-frame p50/p95, CPU enqueue/join,
   actual queue submissions, upload/readback bytes, GPU stage times,
   memory placement and committed arena capacity. Broadphase placement
   comparisons vary motion fraction and pair churn separately from body count.
4. Vulkan and D3D12 physical correctness and performance qualification;
   WARP and software Vulkan are functional checks, not performance evidence.
5. Explicit retirement of any redundant source-stream/transfer plumbing
   replaced by the Scene-owned graph. A temporary coexistence must not be
   released as the final execution architecture.

The initial shared-frame input applies to the Sc shape-transform producer.
Its input allocation is the device-bound source record on supported
hardware; no GPU upload command or copy is emitted for those bytes.
