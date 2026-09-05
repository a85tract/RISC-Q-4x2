"""The SYNC-mode LMX2594 list for 491.52 MHz out of 245.76 MHz: xrfclk's LMX2594_491.52.txt with
R0[14] VCO_PHASE_SYNC = 1 and PLL_N 320 -> 80 (IncludedDivide 4: CHDIV 16 = SEG0 x SEG1 x ..., datasheet
SNAS696C 7.3.10.3 step 4). Everything else unchanged. Writes argv[2] from argv[1]."""
import sys

src, dst = sys.argv[1], sys.argv[2]
out, seen = [], {}
for line in open(src, encoding="utf-8"):
    line = line.rstrip("\r\n")
    if not line.strip():
        continue
    name, val = line.split()
    v = int(val, 16)
    addr, data = v >> 16, v & 0xFFFF
    seen[addr] = data
    if addr == 0:
        assert data == 0x241C, hex(data)
        data |= 1 << 14                   # VCO_PHASE_SYNC
    if addr == 36:
        assert data == 320, data          # PLL_N
        data = 80
    out.append(f"{name}\t0x{addr:02X}{data:04X}")
assert seen[75] >> 6 & 0x1F == 5, "CHDIV code 5 (/16) expected"
assert seen[34] == 0 and seen[44] & 7 == 0 and seen[42] == 0 and seen[43] == 0, "integer mode, N < 2^16"
assert seen[31] >> 14 & 1 == 1, "SEG1_EN"
open(dst, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print(f"wrote {dst}: {len(out)} registers; R0 0x{seen[0] | 1 << 14:04X}, R36 {80}")
