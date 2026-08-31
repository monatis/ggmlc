"""C++ Project Code Generation for GGML Dialect Graphs."""

import sys
import types

from ggmlc.codegen.cpp import GGMLCCppCodeGenerator, generate_cpp_project


class _CodegenModule(types.ModuleType):
    """Makes the ggmlc.codegen subpackage directly callable as ggmlc.codegen(...)."""

    def __call__(self, *args, **kwargs):
        from ggmlc.compiler import codegen

        return codegen(*args, **kwargs)


sys.modules[__name__].__class__ = _CodegenModule

__all__ = [
    "GGMLCCppCodeGenerator",
    "generate_cpp_project",
]
