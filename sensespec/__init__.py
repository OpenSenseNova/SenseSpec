# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""SenseSpec - Torch native spec decode training framework."""

from importlib import import_module

# Keep the package initializer free of CUDA imports so runtime setup can run
# before torch/vLLM/Mooncake are loaded. Public model exports remain unchanged.
_EXPORTS = {
    "Eagle3Model": "sensespec.models",
    "DFlashModel": "sensespec.models.dflash",
    "DFlashConfig": "sensespec.models.draft.dflash",
    "DFlashDraftModel": "sensespec.models.draft.dflash",
    "DFlash2Model": "sensespec.models.dflash2",
    "DFlash2Config": "sensespec.models.draft.dflash2",
    "DFlash2DraftModel": "sensespec.models.draft.dflash2",
    "DSparkModel": "sensespec.models.dspark",
    "DSparkConfig": "sensespec.models.draft.dspark",
    "DSparkDraftModel": "sensespec.models.draft.dspark",
    "AutoDraftModelConfig": "sensespec.models.draft",
    "AutoEagle3DraftModel": "sensespec.models.draft",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
