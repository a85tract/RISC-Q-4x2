"""riscq.lang — the @kernel language (spec 02): a typed python subset compiled to C."""

from riscq.lang.kernel import (Array, Kernel, KernelCompileError, ParamTable, compile_kernel,
                               kernel)

__all__ = ["Array", "Kernel", "KernelCompileError", "ParamTable", "compile_kernel", "kernel"]
