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
  Dy's contact packer shares a single inline body input per immovable
  actor across the frame's managers; responding bodies use resident solver
  inputs instead of repeating their state in contact descriptors.
- **Device numerical state:** broadphase scratch, PCM witnesses, contact
  history, island worklists, constraint rows, and solver motion remain in
  GPU-preferred storage. Dy packs authored integration inputs separately
  from its per-solver-body world inertia matrices; native device-owned
  inertia is computed into GPU work storage without host tensor uploads.
  Stable Sc joint-source slots reside apart from EdgeIndex friction history;
  PCM history growth preserves joint residency without reuploading unchanged
  definitions. Sc queues changed joint-source records before the PCM witness
  join; subsequent Dy preparation consumes the same resident slots without
  uploading them again. Dependent passes exchange device addresses or offsets;
  they do not materialize host mirrors between passes.
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

Finite-pair GJK uses the dense one-dispatch continuation on native Vulkan
and D3D12 hardware. D3D12 software adapters run the staged continuation
for API correctness; their dispatch count and wall time are not the hardware
execution plan. Both paths produce the same Sc EdgeIndex contact stream.


The Scene assigns one epoch to authored input, device work, and public
publication. The GPU writes transient results into epoch-owned work
resources. Persistent histories use a single owner and an explicit
commit/undo boundary. A failed epoch cannot expose speculative contacts,
motion, or events through the public Scene.

The current production path joins SAP found/lost reports for Sc admission
and compact GPU PCM headers for CPU touch and friction correlation. The
producer publishes contiguous headers alongside its resident contact stream
without another dispatch. The same producer validates source identity,
contact counts, resident transforms, normals, and world witnesses before
publishing physical touch. Failed PCM publication marks the journaled
EdgeIndex touch word; device island admission rejects that epoch without
waiting for Sc to read its headers. Sc consumes valid headers before Dy
admission, while full world-point readback for physically touching pairs
joins the existing Dy completion transfer. Zero-contact pairs transfer their header
without a second witness download. Collision-only publication and frames
with no numerical Dy task fetch required points at their public consumer. GPU PCM numerical admission uses one
Sc friction patch per manager and advances contact ranges without walking
device-owned contact points; CPU Gu retains its general multi-patch correlation. Portable GPU worlds with PCM and Dy enable Sc device island admission.
Pre-witness Sc graphs retain device-authored tasks for eligible compact
islands. Late admission is selected when Sc task batching hides independent
responding islands; wide numerical groups use device partitioning instead.
Selected admission passes stable contact and joint identities, reads resident
physical touch, and supplies task/body ranges without host task/ordinal
uploads. Every Scene-authored GPU contact descriptor and device response token
addresses the original Sc pair-candidate ordinal. Physically non-touching
candidates retain empty numerical slots; CPU PGS alone indexes published
Gu manifolds. The island work and Dy solve enter one ordered GPU
submission without an intermediate host wait. Dy reads Sc's resident
body-ordinal and source response-token streams directly. Dy updates the transient Sc task records directly and reads the
admission header in place. One Sc work-buffer binding serves the admission header, tasks,
body ordinals and response tokens without a device copy; descriptor
coloring stays in Dy's numerical scratch. Island graphs
with at most 128 nodes and 128 edges build their task records in one dispatch;
larger graphs use parallel union, prefix and response ordering.
Wide islands use their separate device-partition execution path; the compact
admission selection rejects them. Sc otherwise reuses CPU-prepared
accurate-island task boundaries. Independent one-constraint islands with one
or two responding bodies execute in packed GPU lanes; dependent compact
islands use fused workgroups. The compact numerical launch uses a two-axis
GPU-authored indirect grid when one dispatch dimension cannot hold its
workgroups; Dy flattens that grid into canonical task ordinals without host
task reconstruction. Device-generated roots are numerical task IDs,
not replacements for Sc's persistent island identities. The Scene prepares kinematic target velocities after PCM submission,
before Sc joins its witness stream; these changes remain covered by the
frame's motion journal. Solver-body admission still follows accurate island
reconciliation. For closed PCM contact and connected 1D joint graphs, Sc compacts
responding ActorIDs into frame-local node ordinals and submits accurate
island admission with stable EdgeIndices and source ordinals before the
host GPU witness join. When bounded CPU primitive Gu
produces some PCM records, admission is queued after those host records
publish physical touch into the same device journal. Unrelated scene
actors never enlarge the resident numerical graph. Early responding nodes
are bounded by the device task capacity rather than the single-workgroup
single-workgroup threshold. Larger source graphs use parallel device admission.
Hardware scene qualification now compares identical CPU, GPU Sc-authored
and GPU device-admitted numerical workloads, including frame percentiles
and CPU-referenced pose/velocity error.
The GPU resolves physical touch and runnable task topology on its own
ordered queue. Sc supplies stable edge and node inputs from one mapped
GPU-local frame block on SharedLocal hardware. Other devices stage the
same bounded input spans on the existing queue. Device output and
persistent EdgeIndex touch history retain their separate owners.
Device admission assigns exact source-ordered contact-row
prefixes from resident PCM and Gu extents. Compact graphs compute the prefix
in their admission workgroup; larger graphs scan response blocks in parallel
using the completed radix workspace. Device-admitted Dy tasks place
PCM and CPU Gu managers directly in those ranges, without repeating
a task-local capacity scan or allocating contact rows with global atomics.
The CPU-authored task path retains its original source-order correlation.
Early admission uses compact Sc node ordinals as Dy's responding body
addresses. Kinematic joint endpoints occupy separate nonresponding slots.
The GPU's responding body order and exact contact-row prefixes come from
device-authored island work. Sc correlates numerical results to ActorIDs
at publication. Late-admitted tasks consume the prepared solver-row order. Device-admitted numerical tasks do
not materialize host endpoint descriptors for physically non-touching PCM candidates;
CPU-authored tasks retain their explicit zero-response endpoints.
Contact-only tasks require an admitted response, so empty PCM candidates
do not produce solver work. Mixed ordinary CPU Gu and GPU PCM contact/joint
graphs also admit before the GPU witness join, after CPU Gu publishes
its EdgeIndex touch/loss journal. Joint tokens address exact row ranges
derived from authored joint kinds before witness publication. Sc defers GPU witness publication
until all such Gu producers complete, regardless of their encounter order.
A single responding dynamic body admits a GPU PCM contact task. Contact
candidates and joint-only islands share the same device graph; physical
touch, not speculative overlap, determines runnable numerical responses.
A lone CPU-origin source edge uses the existing CPU-authored numerical
task rather than launching a device graph and remapping a singleton.
GPU PCM sources retain pre-witness admission from their resident touch
producer, including single-contact scenes.
Sc stamps every PCM request with its original speculative candidate ordinal
before Gu submission. Mixed CPU Gu scheduling, the early device response
graph, and Dy's candidate-indexed manifold and contact-preparation tables
consume that identity directly, including gaps where Gu publishes no contact
manager. Only responding, prepared candidates own a contact-preparation entry;
non-touching and zero-response candidate slots remain invalid. Joint graphs
admit stable Sc edge identities with the same early contact graph. Each
joint token addresses its first source-ordered Dy row directly, independently
of the CPU task order. CPU Sc still owns the committed island graph
and the joint and contact publication lifetimes.
The Scene joins GPU PCM headers for public contact state and
reads back Dy scheduling and numerical results for publication.
Friction candidate validity and retained anchors are prepared in one
source-ordered pass while GPU PCM executes. The contact producer does
not consume those masks; Dy selects retained sources from the same
candidate records without rescanning the prior patch list.
When GPU Sc has already admitted an early numerical island graph, Sc
retains PCM source identities and numerical capacity without allocating
host contact-point slots. CPU Gu generates its own compact contact arena
while device witnesses remain resident. At header publication, Sc expands
the exact PCM ranges backwards, preserving CPU Gu order and public event
lifetimes without copying or compacting speculative PCM point reservations.
GPU contact points remain deferred until their public consumer.
Other PCM frames retain their single-pass publication. Host-generated-only
PCM snapshots and Gu records enter one native submission when the host
witnesses are ready. Rejecting an unsubmitted snapshot discards its queued
commands without restoring nonexistent device journal data; submitted
epochs restore the original EdgeIndex history on rollback. Mixed late-Gu
frames queue the CPU-generated physical-touch and loss journal before
joining PCM's host witness, without waiting on that later device continuation.
During GPU SAP, Sc derives kinematic target velocities from authored
poses and the timestep, then stages actor-local mass, pose and sleep
inputs for the awake accurate-island actor set, including free-flight
bodies. Kinematic endpoints of responding retained contacts also stage
their resulting velocities before Sc observes SAP or PCM reports. GPU
PCM subsequently stages newly intersecting actors that were absent
from the preceding accurate graph, before the host witness join and
even when early device island admission is unavailable. Speculative PCM endpoints stage before
contact classification when the authored pair permits physical response.
A response-disabled PCM pair does not stage numerical endpoints, including
immovable actors; the final Dy pass also omits immovable participants
without contact patches. CPU Gu immovable endpoints stage only after
CPU narrowphase reports a touching, responding manager, while PCM is
still in flight. Empty Gu candidates never upload actor-local inputs.
Already-active bodies and PCM candidate endpoints supply stable actor IDs;
physical touch remains device-authored.
Numerical enqueue reconciles final Sc changes and transfers only changed
actor slots. The actor-local GPU pool remains the sole motion input owner.
First admission initializes contiguous actor records; later updates transfer
only changed mass/dynamics, immovable-velocity, sleep and pose ranges.
Adjacent ranges share an ordered upload, and host pose or mass edits do not
overwrite GPU-owned dynamic velocities. Both pre-witness and final Dy
admission use this same field-addressed transfer policy.

GPU 1D constraints author their numerical rows and joint geometry in
the device input stream. The CPU PGS JointRow arena is not reserved or
populated on GPU frames. Native GPU world motion reads inverse mass from
the actor-local integration input, not a second Sc-authored body array;
the transient body span only provides the Dy result count. Contact-manager
inputs carry tagged references to integrated solver rows or the stable
actor-local pool. Static and kinematic contact participants retain their
authored velocities and mass-frame transforms in the same actor-local
allocation before PCM witness publication. Sc stages responding immovable
contact endpoints once per collision epoch, without rescanning prepared
managers at numerical submission or transferring repeated inline bodies.
Scene emits an 80-byte compact contact source containing actor IDs, prepared
solver ordinals, inverse masses and the numerical material tail. Material and actor
identities retain original Sc candidate order; PCM runs independently
while Sc combines numerical materials and validates friction history.
Collision-only publication omits numerical contact preparation, but
checks material eligibility for retained friction history so the next
numerical frame can reuse those anchors. After accurate island admission, Sc fills solver ordinals
and CPU Gu offsets, retires unused PCM preparation slots, and uploads
only responding sources. Material, friction-history candidates and contact
preparation share one sparse source-index map, preserving Sc candidate
identities without allocating records for speculative Gu holes. Dy lowers the compact result
to 48-byte per-manager GPU descriptors and one shared 16-byte frame
coefficient header. The numerical producer borrows source PCM identities,
EdgeIndex handles, and CPU Gu offsets from those inputs for the duration
of the queued solve; it does not copy a second host provenance table.
The 192-byte inline-body prefix and associated CPU packing are not
constructed on production Scene frames. Standalone numerical inputs without
a Scene actor identity retain their explicit inline body payload.
All numerical contacts in one frame share inverse-timestep, timestep,
rigid-bias and bounce-threshold coefficients.
The GPU Dy path still fetches 8 bytes of edge/touch metadata for ordinary
GPU pairs before numerical admission. Sc validates geometrically sound,
material-compatible retained friction patches while GPU PCM executes.
When more than one historical patch is eligible, Sc queues their
source-ordered local normals and anchors into Dy's resident friction input
before the host PCM witness join. Supported SharedLocal adapters write the
small source directly into GPU-local CPU-mappable storage; other adapters
stage it on the existing ordered queue. Dy reads the same typed source
during numerical preparation, without repacking or re-uploading it after
the join. Standalone numerical inputs transfer only selected candidates.
Dy chooses the first compatible historical normal against the resident
PCM normal and reports the selected source ordinal through the existing
patch result. Pre-witness transport includes all eligible historical
candidates, including pairs later found non-touching; the solver consumes
only the admitted numerical sources.
One eligible cached patch retains its direct device normal check and
regenerates anchors when the normal changes. Broken, retired or
material-incompatible history does not require a normal transfer.
This removes the separate early normal readback at the cost of bounded
candidate uploads; representative hardware latency and transfer-cost
qualification is still required. Full PCM-to-Dy numerical submission
continues to depend on Sc's host header publication. Resident PCM supplies
numerical normals and contacts directly to Dy. The Sc EdgeIndex touch word retains exact PCM
emission counts in bits 8–10, with bit 16 marking the PCM source. CPU Gu
uses bits 8–15 for its exact count and bit 17 for its numerical source;
both paths preserve the existing eight-byte Gu touch upload and journal.
Bit 0 still records physical touch, and bit 31 rejects a failed producer.
Sc accurate island admission writes these counts beside its source-ordered
response tokens. CPU-generated Gu and GPU-generated PCM contacts use the
same device-admitted response stream for selected compact numerical tasks.
When no PCM pair is active, the existing Gu touch journal opens the resident
EdgeIndex epoch without allocating a parallel contact registry. CPU Gu-only
admission follows its touch publication before host numerical task lowering.
Joint-only islands admit directly from typed Sc constraint edges without
creating PCM contact history or running narrowphase; source joint rows
remain the authoritative solver inputs. Dy rejects missing or mismatched PCM and CPU Gu counts before
constructing solver rows, with no extra dispatch or host join.
Legacy CPU Gu responses keep their host-authored extents. For
device-admitted single-patch PCM managers,
Sc supplies a reserved contact and friction window, but no numerical
contact count. Sc preserves a stable source-indexed impulse reservation while Dy
resolves the numerical row base, exact emitted count, and direct patch
interval in device memory. The two row-address domains are separate;
GPU force writeback maps numerical points to their Sc source slots.
Device-admitted tasks allocate disjoint exact-size working ranges from
one GPU-owned numerical row counter. Within each task, PCM and CPU Gu
response counts are prefixed in contact order; interleaved source
reservations do not constrain numerical compaction. The parallel Sc
source-order prefix uses one device dispatch for at most 128 edges
and the existing four-phase scan for larger response streams; no host
scan or additional numerical dispatch is required. Packed singleton
tasks use the same allocator. GPU force writeback retains Sc's source
offsets. Allocation and compaction need no additional dispatch or host join.
Dy returns the exact contact count in its existing four-byte patch report
(bit 30 valid, bits 16–18 count), alongside friction feedback. The host
publishes only the active force slots and leaves unused reservation slots
untouched. CPU Gu retains its correlated source extents; no additional
readback or numerical registry is required. A deferred span transfers 76 bytes for its
first touching pair, or 64 bytes when Sc already fetched that normal;
subsequent pairs in the same span contribute 84 bytes each. Transfers
join Dy's existing completion batch without another point-fetch fence
on responding frames. A completed Dy numerical fence retires preceding
source uploads and PCM readback submissions on the shared queue; PCM witness
bytes remain unpublished until Sc joins or discards that readback.
Compact device island graphs with up to 256 nodes and 256 edges use
one workgroup admission instead of the parallel prefix and sort schedule.
Larger graphs keep source-ordered exact contact-row offsets on the GPU:
up to 128 edges use one block scan, 129–512 edges use one bounded scan,
and larger response streams use parallel block scans with group carries
only beyond 16,384 edges;
257-entry graphs retain the parallel path.
Sc leaves device-authored body and typed response order intact.
Early device-admitted tasks index Dy body and contact preparation directly
by their pre-witness source ordinals, regardless of the CPU's solver-row
order. Sc publishes interaction-ordered contact endpoint ordinals in its
resident source-indexed admission table. Host contact preparations carry
unresolved dynamic-body references, not a duplicate source-node map.
A separate Dy preparation dispatch validates active body inputs and
resolves only admitted references from the resident table on the
same command queue, before PGS expands responding descriptors.
Host-scheduled tasks consume their own prepared row indices.
Joint source rows occupy exact contiguous ranges in descending constraint
identity order. The device's typed joint tokens address those ranges
directly. Early Sc admission publishes indirect launch arguments without
a numerical body or joint remap upload or dispatch.
Sc reserves a source-ordered numerical input table independently of PCM
touch classification. GPU PCM candidates receive typed actor endpoints
and material coefficients while the contact producer executes, before
Dy admission. CPU Gu candidates receive numerical inputs only when
their local narrowphase produces contacts,
while PCM is still running. Untouched Gu candidates do not acquire
numerical preparation records. Dy consumes the resulting source-indexed
table without repeating material combination. Contact/friction row
construction still follows Sc's witness publication.
Early device admission reserves bounded PCM normal-force and friction
anchor windows by source candidate before witness publication. Non-touching
candidates occupy no numerical task; the device assigns exact contact working
rows from its admitted response prefixes. CPU Gu appends its correlated
rows after the PCM source capacity. Joint numerical descriptors and their
kinematic endpoint slots are also prepared before the witness join.
PCM numerical patch descriptors follow the original producer ordinal,
independently of the compact public friction cache. CPU Gu appends its
correlated patch descriptors. Public writeback resolves the numerical
patch through its candidate manager. Candidate dynamic actor inputs,
including dormant candidates, occupy their final integration ordinals
before the witness join. Sc finalizes integration eligibility and wake
energy after island activation; it does not reconstruct the candidate
mass frames, dynamics or force inputs. Counted interactions are already
fixed by pair and constraint admission. Source-indexed dynamics integrate
at the same ordinal whether their component admits a solver task or free
flight. Response-disabled contacts retain both dynamic graph endpoints, but
publish no response token. Bodies connected to a responding component
share its iteration and integration range; components without responses
remain free-flight work. The source-indexed result path does not download
the redundant solver-body ordinal list. Unrelated free actors occupy
the suffix. The early device path retains the PCM preparation prefix
through publication; host-scheduled
tasks compact their preparation by touching manager. Numerical enqueue
still requires Sc activation, sleep arbitration and free-body admission.
Known authored and SAP wake seeds are projected through the existing
speculative island membership before numerical admission. Dormant
candidates in that wake reach can supply retained contact and joint
response sources; actor wake and event publication remain transactional.
Potential physical wakes use dense representatives of existing
speculative groups. New touches set the group's transient wake bit;
accurate components consume that bit before task admission without
merging their response topology. Fully awake graphs need no wake-group
source words. Accurate components reduce published sleep readiness,
new touch activation and connected active-kinematic vetoes to decide
runnable numerical tasks. A ready member does not suppress the admission
of its still-active component. Compact all-awake scenes with a complete
resident PCM source range submit Dy before the witness join. Independent
awake bodies occupy the existing free-integration suffix, without
entering the accurate candidate graph. Sc publishes the witnesses and
queues sparse PCM point DMA with numerical results, then consumes the
submitted source ordinals. Conditional activation, newly admitted free
sources outside the active inventory, and wide components retain later
authorization.

Retiring this join requires one Sc-authored, source-ordered table of stable
candidate identities and typed responding endpoints. GPU PCM must derive
actual touch, compact contact/constraint ranges, and runnable task extents
from resident results before Dy consumes them. Numerical admission must not
require Sc to reconstruct its current contact rows or friction patches from
the header transfer. Sc reconciles the completed physical-touch transitions,
retained friction history, public events, and persistent island identities
under the same transaction. Mixed CPU Gu and host callbacks retain their
explicit producer and observation dependencies; zero-contact candidates
never acquire numerical response rows. A rejected epoch cannot publish
speculative motion or advance an EdgeIndex history.

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

Shared-frame input applies to Sc shape-transform records and Dy retained
friction sources on supported UMA adapters. The CPU writes directly into
GPU-local storage before its ordered device consumers; no transfer command
or copy is emitted for those bytes. Unsupported placements use the staged
source queue rather than a CPU-only GPU input allocation.
