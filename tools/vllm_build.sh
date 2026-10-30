#!/usr/bin/env bash
# 当前仓库 vLLM 的完整 CUDA 源码构建流程（不修改 vLLM 源码）。
# 目标：Linux x86_64、Python 3.12、PyTorch 2.13.0、CUDA 13.0、SM90（H800/H100）。
# 包括 CUDA/C++ 扩展、DeepGEMM、Rust 前端及工具解析器；不复用已有二进制。
#
# 用法：在仓库根目录执行 bash tools/vllm_build.sh。
# 系统需预装 GCC/G++、Git、curl、tar、uv、protoc，以及运行时所需的 NVIDIA 驱动。
# uv 未安装时可先执行：python3 -m pip install --user uv
# 并确保 uv 所在目录（通常为 ~/.local/bin）在 PATH 中。
#
# 默认使用当前项目 .cache/vllm-build/ 下的专用环境和源码压缩包缓存。
# 也可用 VLLM_BUILD_ENV 指定已有的 Python 3.12 venv/Conda 环境，复用兼容的依赖。
# AFS/NFS 上建议把环境及缓存放到本地磁盘，例如：
#   VLLM_BUILD_ENV=/local/ssd/vllm-py213 \
#   VLLM_SOURCE_CACHE=/local/ssd/vllm-source-cache bash tools/vllm_build.sh
# 可配置：VLLM_SOURCE_DIR、VLLM_BUILD_ENV、VLLM_SOURCE_CACHE、VLLM_LOG_DIR、
#         VLLM_BUILD_TMPDIR、VLLM_CARGO_HOME、VLLM_RUSTUP_HOME、VLLM_UV_BIN、VLLM_PYTHON、
# VLLM_PYTHON 可指定已有 Python 3.12；不设置时 uv 自动查找或下载 3.12。
#         MAX_JOBS（默认 48）、NVCC_THREADS（默认 2）。
# MAX_JOBS 会除以 NVCC_THREADS，默认实际并行编译任务数为 24。
#
# --prepare-only：只准备环境和全部 CUDA 依赖源码，不删除或重编译现有 .so。
# --verify-only：只验证指定环境中的版本、当前源码路径、原生扩展及 Rust 前端。
# 日志：默认 logs/vllm-build/<模式>-<时间>-<PID>.log，包含 stdout/stderr 和退出码。
# 网络：沿用调用者的 HTTPS_PROXY/https_proxy、NO_PROXY/no_proxy，不修改全局代理。
# 如代理不可用且允许直连，可设置 NO_PROXY=codeload.github.com 再执行。
#
# 每次构建均使用全新目录，禁用编译缓存和预编译 vLLM wheel；只缓存依赖源码包。
# 保留必要的 PyTorch/CUDA 基础依赖；--no-deps 将构建依赖和模型运行依赖分开。
# 这是 SM90（H800/H100）构建环境，不自动安装完整模型推理依赖。
set -euo pipefail

usage() {
  printf '%s\n' \
    '用法：bash tools/vllm_build.sh [--prepare-only | --verify-only | --help]' \
    '默认：准备独立环境 → 获取固定版本源码 → 全量编译安装 → 验证原生扩展。' \
    '环境、缓存、并行数及日志路径的配置示例见脚本开头。'
}

die() { printf '错误：%s\n' "$*" >&2; exit 1; }

require_pin() {
  grep -Fq -- "$2" "$1" || die "源码依赖版本已变化，请更新脚本中的固定版本：$1"
}

fetch_source() {
  local repo=$1 revision=$2 destination=$3 archive temporary
  archive="$SOURCE_CACHE/${repo//\//-}-$revision.tar.gz"
  if [[ ! -f "$archive" ]]; then
    temporary=$(mktemp "$SOURCE_CACHE/.download.XXXXXX")
    printf '下载源码：%s @ %s\n' "$repo" "$revision"
    if ! curl --fail --location --silent --show-error \
      --retry 3 --retry-all-errors --connect-timeout 20 \
      --speed-limit 1024 --speed-time 60 \
      "https://codeload.github.com/$repo/tar.gz/$revision" -o "$temporary"; then
      rm -- "$temporary"
      return 1
    fi
    if ! tar -tzf "$temporary" >/dev/null; then
      rm -- "$temporary"
      return 1
    fi
    mv -- "$temporary" "$archive"
  else
    printf '使用源码压缩包缓存：%s @ %s\n' "$repo" "$revision"
  fi
  mkdir -p "$destination"
  tar -xzf "$archive" --strip-components=1 -C "$destination"
}

finalize_build() {
  "$BUILD_ENV/bin/python" -I - "$SOURCE_DIR" "$LOG_FILE" <<'PY'
from pathlib import Path
import re
import shutil
import sys
import sysconfig

source = Path(sys.argv[1]).resolve()
# 旧 DeepSpec 环境可能另有一个 .pth，它会抢在 editable finder 前导入旧源码。
path_file = Path(sysconfig.get_path('purelib')) / 'deepspec-vllm-source.pth'
if path_file.exists() and path_file.read_text() != str(source) + '\n':
    shutil.copy2(path_file, sys.argv[2] + '.pth.before')
    path_file.write_text(str(source) + '\n')
    print('已更新环境中的 vLLM 源码路径:', path_file)

# editable 安装可能复制而非保留 cute/ 链接；两种情况都要重写绝对包导入。
destination = source / 'vllm/vllm_flash_attn/cute'
if destination.is_symlink():
    original = destination.resolve(strict=True)
    destination.unlink()
    shutil.copytree(original, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for path in destination.rglob('*.py'):
    original = path.read_text()
    updated = re.sub(r'(?<![\w.])flash_attn\.cute', 'vllm.vllm_flash_attn.cute', original)
    if updated != original:
        path.write_text(updated)
print('FA4 Python 文件已本地化:', destination)
PY
}

verify_build() {
  "$BUILD_ENV/bin/python" -I - "$SOURCE_DIR" "$BUILD_ENV" <<'PY'
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import torch
import vllm

source = Path(sys.argv[1]).resolve() / 'vllm'
environment = Path(sys.argv[2]).resolve()
assert Path(sys.prefix).resolve() == environment, sys.prefix
assert Path(torch.__file__).resolve().is_relative_to(environment), torch.__file__
assert torch.__version__.split('+')[0] == '2.13.0', torch.__version__
assert torch.version.cuda == '13.0', torch.version.cuda
assert Path(vllm.__file__).resolve().parent == source, vllm.__file__
print('Python:', sys.executable)
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('vLLM:', vllm.__version__, vllm.__file__)
for name in (
    'vllm._C_stable_libtorch', 'vllm._moe_C_stable_libtorch',
    'vllm.cumem_allocator', 'vllm.spinloop', 'vllm.fs_io_C',
    'vllm._flashkda_C', 'vllm._rust_tool_parser',
    'vllm._flashmla_C', 'vllm._flashmla_extension_C',
    'vllm.vllm_flash_attn._vllm_fa2_C',
    'vllm.vllm_flash_attn._vllm_fa3_C',
):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(source), path
    print('扩展加载通过:', name, path)
# 独立加载 DeepGEMM 原生扩展，避免要求完整推理依赖。
deepgemm = list((source / 'third_party/deep_gemm').glob('_C.cpython-312-*.so'))
assert len(deepgemm) == 1, deepgemm
spec = importlib.util.spec_from_file_location('_C', deepgemm[0])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print('扩展加载通过: DeepGEMM', deepgemm[0])
assert not (source / 'vllm_flash_attn/cute').is_symlink(), 'FA4 仍依赖外部临时目录'
subprocess.run([str(source / 'vllm-rs'), '--help'], check=True, stdout=subprocess.DEVNULL)
print('Rust 前端启动通过:', source / 'vllm-rs')
print('验证通过：PyTorch 2.13 / CUDA 13，12 个扩展及 Rust 前端均来自指定源码目录。')
PY
}

main() {
  local mode=build uv_bin command_name dependency revision destination artifact relative rust_toolchain
  [[ $# -le 1 ]] || { usage; return 2; }
  case "${1:-}" in
    '') ;;
    --prepare-only) mode=prepare ;;
    --verify-only) mode=verify ;;
    -h|--help) usage; return 0 ;;
    *) usage; return 2 ;;
  esac

  SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
  SOURCE_DIR=$(realpath -m -- "${VLLM_SOURCE_DIR:-$PROJECT_ROOT/vllm4spec}")
  BUILD_ENV=$(realpath -m -- "${VLLM_BUILD_ENV:-$PROJECT_ROOT/.cache/vllm-build/venv}")
  SOURCE_CACHE=$(realpath -m -- "${VLLM_SOURCE_CACHE:-$PROJECT_ROOT/.cache/vllm-build/source-archives}")
  LOG_DIR=$(realpath -m -- "${VLLM_LOG_DIR:-$PROJECT_ROOT/logs/vllm-build}")
  BUILD_STARTED_SECONDS=$SECONDS
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/$mode-$(date +%Y%m%d-%H%M%S)-$$.log"
  exec > >(tee "$LOG_FILE") 2>&1
  trap 'status=$?; printf "\n结束时间：%s；退出码：%s；耗时：%s 秒\n日志：%s\n" "$(date -Is)" "$status" "$((SECONDS - BUILD_STARTED_SECONDS))" "$LOG_FILE"; exit "$status"' EXIT
  printf '开始时间：%s\n模式：%s\n源码：%s\n环境：%s\n日志：%s\n' \
    "$(date -Is)" "$mode" "$SOURCE_DIR" "$BUILD_ENV" "$LOG_FILE"
  [[ -f "$SOURCE_DIR/setup.py" && -f "$SOURCE_DIR/CMakeLists.txt" ]] || \
    die '找不到 vLLM 源码；请初始化仓库根目录的 vllm4spec/ 子模块，或设置 VLLM_SOURCE_DIR。'

  # 先保留 uv 的绝对路径，再排除调用者的 Conda、Python 和原生库路径。
  uv_bin=${VLLM_UV_BIN:-$(command -v uv || true)}
  if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then uv_bin="$HOME/.local/bin/uv"; fi
  [[ -z "$uv_bin" ]] || uv_bin=$(realpath -- "$uv_bin")
  unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV
  unset CMAKE_PREFIX_PATH CMAKE_ARGS CUDA_HOME CUDA_PATH LD_LIBRARY_PATH
  unset VLLM_CUTLASS_SRC_DIR VLLM_FLASH_ATTN_SRC_DIR FLASH_MLA_SRC_DIR
  unset QUTLASS_SRC_DIR DEEPGEMM_SRC_DIR FLASH_KDA_SRC_DIR FMHA_SM100_SRC_DIR
  unset TML_FA4_SRC_DIR TRITON_KERNELS_SRC_DIR
  unset VLLM_USE_PRECOMPILED VLLM_USE_PRECOMPILED_RUST VLLM_PRECOMPILED_WHEEL_LOCATION
  unset VLLM_DOCKER_BUILD_CONTEXT VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL
  export PYTHONNOUSERSITE=1
  export PATH="$BUILD_ENV/bin:/usr/local/bin:/usr/bin:/bin"

  if [[ "$mode" == verify ]]; then
    [[ -x "$BUILD_ENV/bin/python" ]] || die '构建环境不存在，请先执行完整构建。'
    BUILD_SITE=$("$BUILD_ENV/bin/python" -I -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
    export LD_LIBRARY_PATH="$BUILD_SITE/nvidia/cu13/lib:$BUILD_SITE/torch/lib"
    verify_build
    return
  fi
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die '本脚本针对 Linux x86_64。'
  [[ -n "$uv_bin" ]] || die '请先安装 uv 并加入 PATH；参见脚本开头。'
  for command_name in gcc g++ git curl tar; do
    command -v "$command_name" >/dev/null || die "缺少系统工具：$command_name"
  done
  require_pin "$SOURCE_DIR/CMakeLists.txt" 'set(CUTLASS_REVISION "v4.4.2"'
  require_pin "$SOURCE_DIR/cmake/external_projects/flashmla.cmake" 'GIT_TAG 8447acbcb558db892bf7c1197d225be1c95b168c'
  require_pin "$SOURCE_DIR/cmake/external_projects/qutlass.cmake" '"e74319e3405ce6d71965732880f5dc1f52371f64"'
  require_pin "$SOURCE_DIR/cmake/external_projects/vllm_flash_attn.cmake" 'GIT_TAG 2b84100f50e1d2a8726a86c86d14c2f9c9e5a67c'
  require_pin "$SOURCE_DIR/cmake/external_projects/deepgemm.cmake" '"8b1392b978f5a03c828dd1711090d7fb50958b8a"'
  require_pin "$SOURCE_DIR/cmake/external_projects/flashkda.cmake" 'GIT_TAG b5d11010ff01c1d4a683c0dde42e76cbeaa8107f'
  require_pin "$SOURCE_DIR/cmake/external_projects/fmha_sm100.cmake" 'GIT_TAG 087c161814d4d9c735b46c21212a09e5f8eb92fa'
  require_pin "$SOURCE_DIR/cmake/external_projects/tml_fa4.cmake" 'GIT_TAG b206834606ed5b5f21f8eed6b0683f528ea9cf7d'
  require_pin "$SOURCE_DIR/cmake/external_projects/triton_kernels.cmake" 'set(TRITON_KERNELS_TAG "v3.5.1")'
  export MAX_JOBS="${MAX_JOBS:-48}" NVCC_THREADS="${NVCC_THREADS:-2}"
  [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ && "$NVCC_THREADS" =~ ^[1-9][0-9]*$ ]] || \
    die 'MAX_JOBS 和 NVCC_THREADS 必须为正整数。'
  export TORCH_CUDA_ARCH_LIST=9.0 VLLM_TARGET_DEVICE=cuda CMAKE_BUILD_TYPE=Release
  export VLLM_DISABLE_SCCACHE=1 CCACHE_DISABLE=1 SETUPTOOLS_SCM_IGNORE_DUBIOUS_OWNER=1

  printf '\n准备构建环境（复用兼容依赖，按源码要求补齐缺失依赖）\n'
  if [[ ! -x "$BUILD_ENV/bin/python" ]]; then
    "$uv_bin" venv --python "${VLLM_PYTHON:-3.12}" "$BUILD_ENV"
  fi
  [[ -f "$BUILD_ENV/pyvenv.cfg" || -d "$BUILD_ENV/conda-meta" ]] || \
    die 'VLLM_BUILD_ENV 必须是 venv 或 Conda 环境，不能指向系统 Python。'
  "$BUILD_ENV/bin/python" -I - "$BUILD_ENV" <<'PY'
from pathlib import Path
import sys
assert sys.version_info[:2] == (3, 12), sys.version
assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve(), sys.prefix
config = Path(sys.prefix) / 'pyvenv.cfg'
if config.exists():
    assert 'include-system-site-packages = false' in config.read_text().lower()
PY
  "$uv_bin" pip install --python "$BUILD_ENV/bin/python" \
    'torch==2.13.0' 'cuda-toolkit[nvcc,cccl]==13.0.3.0' \
    'numpy==2.3.5' -r "$SOURCE_DIR/requirements/build/cuda.txt"
  "$uv_bin" pip freeze --python "$BUILD_ENV/bin/python" > "${LOG_FILE%.log}-packages.txt"
  BUILD_SITE=$("$BUILD_ENV/bin/python" -I -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
  export CUDA_HOME="$BUILD_SITE/nvidia/cu13" CUDA_PATH="$BUILD_SITE/nvidia/cu13"
  # CUDA wheel 提供版本化库，但缺少 CMake/nvcc 查找所需的链接名和 lib64 目录。
  for dependency in libcudart libnvrtc; do
    [[ -f "$CUDA_HOME/lib/$dependency.so.13" ]] || die "缺少 CUDA 库：$dependency.so.13"
    if [[ ! -e "$CUDA_HOME/lib/$dependency.so" ]]; then
      ln -s "$dependency.so.13" "$CUDA_HOME/lib/$dependency.so"
    fi
  done
  [[ -e "$CUDA_HOME/lib64" ]] || ln -s lib "$CUDA_HOME/lib64"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:$BUILD_SITE/torch/lib"
  "$BUILD_ENV/bin/python" -I - <<'PY'
import torch
assert torch.__version__.split('+')[0] == '2.13.0', torch.__version__
assert torch.version.cuda == '13.0', torch.version.cuda
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, '路径:', torch.__file__)
PY
  gcc --version
  nvcc --version
  "$uv_bin" --version

  # 全新工作目录；压缩包只包含源码，避免 Git 历史下载及 AFS 子模块所有权问题。
  BUILD_TMP_ROOT=$(realpath -m -- "${VLLM_BUILD_TMPDIR:-${TMPDIR:-/tmp}}")
  mkdir -p "$BUILD_TMP_ROOT" "$SOURCE_CACHE"
  BUILD_DIR=$(mktemp -d "$BUILD_TMP_ROOT/vllm-build.XXXXXX")
  NATIVE_SOURCES="$BUILD_DIR/sources"
  export FETCHCONTENT_BASE_DIR="$BUILD_DIR/deps" TMPDIR="$BUILD_DIR/tmp"
  mkdir -p "$TMPDIR"
  # 仅本次构建信任指定源码，兼容共享文件系统的不同 UID，不改用户全局 Git 配置。
  if [[ -f "${GIT_CONFIG_GLOBAL:-$HOME/.gitconfig}" ]]; then
    git config --file "$BUILD_DIR/gitconfig" --add include.path "${GIT_CONFIG_GLOBAL:-$HOME/.gitconfig}"
  fi
  git config --file "$BUILD_DIR/gitconfig" --add safe.directory "$SOURCE_DIR"
  export GIT_CONFIG_GLOBAL="$BUILD_DIR/gitconfig"
  if [[ -x "$BUILD_ENV/cargo/bin/rustup" && -d "$BUILD_ENV/rustup" ]]; then
    export CARGO_HOME="$(realpath -m -- "${VLLM_CARGO_HOME:-$BUILD_ENV/cargo}")"
    export RUSTUP_HOME="$(realpath -m -- "${VLLM_RUSTUP_HOME:-$BUILD_ENV/rustup}")"
  else
    export CARGO_HOME="$(realpath -m -- "${VLLM_CARGO_HOME:-$PROJECT_ROOT/.cache/vllm-build/cargo}")"
    export RUSTUP_HOME="$(realpath -m -- "${VLLM_RUSTUP_HOME:-$PROJECT_ROOT/.cache/vllm-build/rustup}")"
  fi
  export CARGO_TARGET_DIR="$BUILD_DIR/rust-target" CARGO_BUILD_JOBS="$MAX_JOBS"
  export VLLM_REQUIRE_RUST_FRONTEND=1
  export RUSTC_WRAPPER= RUSTC_WORKSPACE_WRAPPER=
  export PATH="$CARGO_HOME/bin:$PATH"
  rust_toolchain=$("$BUILD_ENV/bin/python" -I -c \
    'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["toolchain"]["channel"])' \
    "$SOURCE_DIR/rust-toolchain.toml")
  if [[ "$rust_toolchain" =~ ^[0-9]+\.[0-9]+$ ]]; then rust_toolchain+=.0; fi
  if [[ ! -x "$CARGO_HOME/bin/rustup" ]]; then
    curl --proto '=https' --tlsv1.2 --fail --location --retry 3 \
      https://sh.rustup.rs -o "$BUILD_DIR/rustup-init.sh"
    sh "$BUILD_DIR/rustup-init.sh" -y --no-modify-path --default-toolchain none --profile minimal
  fi
  if ! rustup run "$rust_toolchain" rustc --version; then
    rustup toolchain install "$rust_toolchain" --profile minimal
  fi
  export RUSTUP_TOOLCHAIN="$rust_toolchain"
  rustc --version
  command -v protoc >/dev/null || die 'Rust 前端需要 protoc；请将 protobuf 编译器安装到构建环境或加入 PATH。'
  cargo fetch --manifest-path "$SOURCE_DIR/rust/Cargo.toml" --locked --target x86_64-unknown-linux-gnu
  printf '\n构建目录：%s\n源码压缩包缓存：%s\n准备固定提交的原生依赖源码\n' "$BUILD_DIR" "$SOURCE_CACHE"
  while read -r dependency revision destination; do
    fetch_source "$dependency" "$revision" "$NATIVE_SOURCES/$destination"
  done <<'SOURCES'
NVIDIA/cutlass v4.4.2 cutlass
JaredforReal/FlashMLA 8447acbcb558db892bf7c1197d225be1c95b168c flashmla
NVIDIA/cutlass 147f5673d0c1c3dcf66f78d677fd647e4a020219 flashmla/csrc/cutlass
IST-DASLab/qutlass e74319e3405ce6d71965732880f5dc1f52371f64 qutlass
NVIDIA/cutlass b2ca083d2bb96c41d9b3c5a930637c641f6669bf qutlass/third_party/cutlass
JaredforReal/flash-attention 2b84100f50e1d2a8726a86c86d14c2f9c9e5a67c flash-attention
NVIDIA/cutlass 62750a2b75c802660e4894434dc55e839f322277 flash-attention/csrc/cutlass
deepseek-ai/DeepGEMM 8b1392b978f5a03c828dd1711090d7fb50958b8a deepgemm
NVIDIA/cutlass f3fde58372d33e9a5650ba7b80fc48b3b49d40c8 deepgemm/third-party/cutlass
fmtlib/fmt 553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28 deepgemm/third-party/fmt
vllm-project/FlashKDA b5d11010ff01c1d4a683c0dde42e76cbeaa8107f flashkda
NVIDIA/cutlass 5c149f52a436782210263fb2f19b354443a61c6a flashkda/cutlass
vllm-project/MSA 087c161814d4d9c735b46c21212a09e5f8eb92fa fmha-sm100
NVIDIA/cutlass eb61c911471867a5fd2466bfd8f29306cea6ebf8 fmha-sm100/python/fmha_sm100/cutlass
vllm-project/tml-fa4 b206834606ed5b5f21f8eed6b0683f528ea9cf7d tml-fa4
triton-lang/triton v3.5.1 triton
SOURCES
  export VLLM_CUTLASS_SRC_DIR="$NATIVE_SOURCES/cutlass" FLASH_MLA_SRC_DIR="$NATIVE_SOURCES/flashmla"
  export QUTLASS_SRC_DIR="$NATIVE_SOURCES/qutlass" VLLM_FLASH_ATTN_SRC_DIR="$NATIVE_SOURCES/flash-attention"
  export DEEPGEMM_SRC_DIR="$NATIVE_SOURCES/deepgemm" FLASH_KDA_SRC_DIR="$NATIVE_SOURCES/flashkda"
  export FMHA_SM100_SRC_DIR="$NATIVE_SOURCES/fmha-sm100" TML_FA4_SRC_DIR="$NATIVE_SOURCES/tml-fa4"
  export TRITON_KERNELS_SRC_DIR="$NATIVE_SOURCES/triton/python/triton_kernels/triton_kernels"
  if [[ "$mode" == prepare ]]; then
    printf '\n环境及依赖源码准备完成，未删除或重编译现有 .so。\n再次执行 bash tools/vllm_build.sh 可进行全量构建。\n'
    return
  fi

  printf '\n开始全量编译：MAX_JOBS=%s，NVCC_THREADS=%s，架构=%s\n' "$MAX_JOBS" "$NVCC_THREADS" "$TORCH_CUDA_ARCH_LIST"
  # 移走旧扩展和 Rust 可执行文件，防止 setup.py 自动复用预编译产物。
  while IFS= read -r -d '' artifact; do
    relative="${artifact#"$SOURCE_DIR/"}"
    mkdir -p "$BUILD_DIR/previous-artifacts/$(dirname -- "$relative")"
    mv -- "$artifact" "$BUILD_DIR/previous-artifacts/$relative"
  done < <(find "$SOURCE_DIR/vllm" -type f \( -name '*.so' -o -name vllm-rs \) -print0)
  printf '旧二进制备份：%s/previous-artifacts\n' "$BUILD_DIR"
  cd -- "$SOURCE_DIR"
  "$uv_bin" pip install --python "$BUILD_ENV/bin/python" \
    --no-cache --no-build-isolation --no-deps --reinstall-package vllm -e . -v
  finalize_build
  printf '\n验证当前目录的编译产物\n'
  verify_build
  printf '\n全量源码构建完成。环境：%s\n完整推理依赖仍需单独安装。\n' "$BUILD_ENV"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
