# syntax=docker/dockerfile:1.7
# RISC-Q 4x2 toolchain images. Build from the REPOSITORY ROOT:
#
#   docker build -f software/client/Dockerfile --target client -t riscq-4x2:client .
#   docker build -f software/client/Dockerfile --target full   -t riscq-4x2:full   .
#
#   client : what every user needs — the RISC-V cross compiler (compiles the kernels that
#            run_experiment/A.run generate) + Python 3.12 with the client packages.
#   full   : client + JDK 17, mill, Verilator, cocotb — the co-simulation (no hardware needed).
#
# Ubuntu 24.04 (noble) on purpose: the prebuilt toolchain tarball is built for it, so lld's
# libxml2/libicu come straight from apt.
ARG TOOLCHAIN_URL=https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2025.01.20/riscv32-elf-ubuntu-24.04-llvm-nightly-2025.01.20-nightly.tar.xz
# Pin after the first build: `sha256sum` of the downloaded tarball. Empty = verify skipped (dev).
ARG TOOLCHAIN_SHA256=

# ── stage 1: the RISC-V toolchain, pruned to what riscq.build.compile_c uses ────────────────────
# compile_c runs `riscv64-unknown-elf-clang -nostdlib -ffreestanding -T link.ld ...` (clang picks
# ld.lld), then objcopy and nm. Kernels are flat images with their own runtime (fw/muldiv.c), so
# no libc/libgcc/gcc backend is needed — the full 3.4 GB tarball prunes to ~250 MB.
FROM ubuntu:24.04 AS toolchain
ARG TOOLCHAIN_URL
ARG TOOLCHAIN_SHA256
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget xz-utils \
 && rm -rf /var/lib/apt/lists/*
RUN wget -q -O /tmp/toolchain.tar.xz "$TOOLCHAIN_URL" \
 && if [ -n "$TOOLCHAIN_SHA256" ]; then echo "$TOOLCHAIN_SHA256  /tmp/toolchain.tar.xz" | sha256sum -c -; fi \
 && mkdir -p /opt && tar -xJf /tmp/toolchain.tar.xz -C /opt && rm /tmp/toolchain.tar.xz \
 && test -x /opt/riscv/bin/clang-19
RUN set -e; mkdir -p /opt/rv/bin /opt/rv/lib/clang /opt/rv/riscv32-unknown-elf \
 && cp -a /opt/riscv/bin/clang-19 /opt/riscv/bin/lld /opt/rv/bin/ \
 && cp -a /opt/riscv/bin/riscv32-unknown-elf-objcopy /opt/riscv/bin/riscv32-unknown-elf-nm /opt/rv/bin/ \
 && cp -a /opt/riscv/lib/clang/19 /opt/rv/lib/clang/ \
 && cp -a /opt/riscv/riscv32-unknown-elf/include /opt/rv/riscv32-unknown-elf/ \
 && cd /opt/rv/bin && ln -s clang-19 clang && ln -s clang-19 riscv32-unknown-elf-clang && ln -s lld ld.lld \
 && for t in clang objcopy nm; do ln -s riscv32-unknown-elf-$t riscv64-unknown-elf-$t; done \
 && rm -rf /opt/riscv && mv /opt/rv /opt/riscv && du -sh /opt/riscv

# ── stage 2: client ──────────────────────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS client
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git libxml2 libicu74 zlib1g libzstd1 liblzma5 libstdc++6 openssh-client rsync \
 && rm -rf /var/lib/apt/lists/*
COPY --from=toolchain /opt/riscv /opt/riscv
# Python 3.12 via uv (cocotb 1.9 does not support 3.14; a pinned, self-contained interpreter)
COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /uvx /usr/local/bin/
RUN uv python install 3.12 && uv venv --python 3.12 /opt/venv312
ENV PATH=/opt/venv312/bin:/opt/riscv/bin:/usr/local/bin:/usr/bin:/bin \
    VIRTUAL_ENV=/opt/venv312 PYTHONDONTWRITEBYTECODE=1
COPY software/client/requirements.txt /tmp/requirements.txt
RUN uv pip install --python /opt/venv312 -r /tmp/requirements.txt
# the repository itself (client package, examples, docs, configs); mount your own clone over
# /work/RISC-Q to develop:  docker run -it --rm -v "$PWD":/work/RISC-Q riscq-4x2:client
WORKDIR /work/RISC-Q
COPY software/client software/client
COPY software/examples software/examples
COPY gateware/configs gateware/configs
COPY docs docs
COPY README.md README.md
ENV PYTHONPATH=/work/RISC-Q/software/client:/work/RISC-Q/sim
# gate: the pruned toolchain must still compile a real kernel (test_build compiles one)
RUN cd software/client && python -m pytest tests/test_build.py tests/test_lang.py -q 2>&1 | tail -1 \
 && python -c "import riscq, numpy, Pyro5; print('client ok')"
CMD ["bash"]

# ── stage 3: full (co-simulation) ────────────────────────────────────────────────────────────────
FROM client AS full
RUN apt-get update && apt-get install -y --no-install-recommends \
      openjdk-17-jdk-headless verilator g++ make curl \
 && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
RUN curl -fsSL https://repo1.maven.org/maven2/com/lihaoyi/mill-dist/1.1.0/mill-dist-1.1.0-mill.sh \
      -o /usr/local/bin/mill && chmod +x /usr/local/bin/mill
COPY sim/requirements.txt /tmp/sim-requirements.txt
RUN uv pip install --python /opt/venv312 -r /tmp/sim-requirements.txt
COPY sim sim
COPY gateware gateware
# a fresh clone without `git submodule update --init --recursive` has empty gateware/ext/*: the RTL
# generator (mill) would fail at first co-sim — fail the image build instead, loudly
RUN test -f gateware/ext/SpinalHDL/build.mill -o -f gateware/ext/SpinalHDL/build.sc \
 || (echo "gateware/ext/SpinalHDL is empty: run 'git submodule update --init --recursive' before building 'full'" && exit 1)
RUN verilator --version && mill --version 2>/dev/null | head -1 \
 && python -c "import cocotb, riscq_sim; print('full ok')"
CMD ["bash"]
