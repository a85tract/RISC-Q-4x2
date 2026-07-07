# Patches to vendored dependencies

`ext/rvls` and `ext/VexiiRiscv` are **git submodules**, so edits inside them are tracked only
by the submodule's own git and are **lost whenever the submodule is reset/updated**. The patches
here capture those edits against the main repo so they can be re-applied.

## `rvls-hart-machine-only-stub-csrs.patch`

**File:** `ext/rvls/src/hart.cpp` (the RVLS↔Spike bridge used for per-instruction lock-step).

**Why:** riscq is a machine-mode-only core. The standard `rv32ui-p-*` startup still pokes a few
S-mode / delegation / PMP CSRs (`satp`, `stvec`, `medeleg`, `mideleg`, `pmp*`) behind the
riscv-tests "set a temporary `mtvec` so an illegal CSR write just skips" guard, and riscq answers
them as write-ignored / read-zero **WARL stubs** instead of trapping. To lock-step against riscq,
Spike is configured **machine-only** (isa `RV32I`, priv `M`) — but Spike keeps those CSRs in its
`csrmap` even without the S extension and **traps** in `verify_permissions` (`!extension_enabled('S')`,
or no PMP). That trap is exactly where riscq commits, so lock-step diverges with *"DUT missed a trap"*.

**What it does:** adds a permissive `stub_csr_t` (a `basic_csr_t` whose `verify_permissions` checks
only the numeric privilege level and skips the extension gate) and, in the `Hart` constructor,
replaces the S-mode/delegation CSRs **only when `!extension_enabled('S')`** and the PMP CSRs **only
when `pmpNum == 0`**. So machine-only Spike (riscq) stops trapping on them, while VexiiRiscv's real
S/U/PMP configs keep their genuine CSRs untouched.

Combined with machine-only `mret` (MPP WARL-clamps to M, so the post-`mret` `ecall` stays cause 11
just like riscq), riscq and Spike stay on the same PC / trap-cause / register stream — no S/U-mode
support and no S/U-mode tests are needed. Result: **all 39 `rv32ui-p-*` pass per-instruction
lock-step**.

### Apply + rebuild

```bash
git -C ext/rvls apply ../../patches/rvls-hart-machine-only-stub-csrs.patch   # re-apply if lost
make -C ext/rvls                                                             # relinks rvls.so (~1s)
```

`make -C ext/rvls` links against the prebuilt Spike static libs in
`ext/VexiiRiscv/ext/riscv-isa-sim/build/`, producing `ext/rvls/build/apps/rvls.so` (the native
library `rvls.jni.Frontend` loads). [`sim/RvTestSim`](../src/riscq/riscv/sim/RvTestSim.scala) auto-enables
lock-step whenever that `.so` exists.
