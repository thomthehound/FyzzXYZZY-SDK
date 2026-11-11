# Scene-owned compute frame

## Execution authority

The native Scene owns one simulation transaction. CPU code owns persistent
actor/shape identity, authored changes, filtering, stable contact-manager
handles, application callbacks, and commit/rollback. The GPU executes dense
numerical work for that transaction. It does not own a second scene,
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
| Broadphase | Preserve stable shape/actor and filter ownership | Bounds, overlap history, found/lost delta counters | Only when identity admission changes |
| Contact | Resolve new manager identities as required | PCM witnesses, persistent touch history, compact event flags | Only for CPU consumers and exceptional contact admission |
| Island and constraint preparation | Author stable joints, body/manager identities and iteration policy | Build accurate numerical components, partitions and rows | None for already admitted topology |
| Dynamics | Own timestep, requested outputs and commit policy | PGS, integration, sleep and force/impulse reductions | Final publication |
| Publish | Validate the completed transaction; update public CPU views | Write compact completion status and requested results | One ordinary completion fence |

A GPU-generated topology delta cannot become an unadmitted contact manager.
An indirect device gate suppresses numerical continuation requiring new
identities; Sc reconciles the compact found/lost stream, then resumes the
same transaction. Stable-topology frames record the full GPU numerical
continuation without that host stop. Contact touch changes still determine
accurate connectivity from the actual GPU contact history, not speculative
broadphase overlap.

Host callbacks, mixed CPU Gu interactions, externally requested body state,
and failure recovery may require additional explicit boundaries. Their
consumers and synchronization edges are part of the frame contract, not
unconditional waits added by individual kernels.

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

The ordinary stable-topology path completes SAP, PCM, island generation,
Dy and numerical writeback in device order before the Scene observes its
required results. GPU-generated indirect arguments supply bounded work
counts; they never authorize new persistent identities.

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
   memory placement and committed arena capacity.
4. Vulkan and D3D12 physical correctness and performance qualification;
   WARP and software Vulkan are functional checks, not performance evidence.
5. Explicit retirement of any redundant source-stream/transfer plumbing
   replaced by the Scene-owned graph. A temporary coexistence must not be
   released as the final execution architecture.

The initial shared-frame input applies to the Sc shape-transform producer.
Its input allocation is the device-bound source record on supported
hardware; no GPU upload command or copy is emitted for those bytes.
