#!/usr/bin/env bash
# shellcheck disable=SC1091
#
# Setup runtime environment based on build_tools/llvm_version.txt

if [ ! "${ASTLROOT}" ] && [ -d /nfs_home/buildkite-slurm/builds/astl ]; then
  source /nfs_home/buildkite-slurm/builds/astl/enable-astl
fi

if [ "${ASTL_LLVM}" ]; then
  # basic utilities functions (git_root, llvm_version)
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/ci/common.sh"

  # LLVM version used to build ASTL-mlir
  if [ ! "${ASTL_LLVM_VERSION}" ] || [ ! -d "${ASTL_LLVM}/${ASTL_LLVM_VERSION}" ]; then
    ASTL_LLVM_VERSION=$(llvm_version)
    if [ "${ASTL_LLVM_VERSION}" ]; then
      export ASTL_LLVM_VERSION;
    fi
  fi

  if [ "${ASTL_LLVM_VERSION}" ]; then
    # setup environment
    export ASTL_LLVM_DIR=${ASTL_LLVM}/${ASTL_LLVM_VERSION}

    # avoid overriding LD_LIBRARY_PATH of initial environment (append)
    if [[ "${LD_LIBRARY_PATH}" != *":${ASTL_LLVM}"* ]]; then
      export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ASTL_LLVM_DIR}/lib
    else
      echo "WARNING: LD_LIBRARY_PATH already refers to ${ASTL_LLVM}!"
    fi
    # avoid overriding PATH of initial environment (append)
    if [[ "${PATH}" != *":${ASTL_LLVM}"* ]]; then
      export PATH=${PATH}:${ASTL_LLVM_DIR}/bin
    else
      echo "WARNING: PATH already refers to ${ASTL_LLVM}!"
    fi

    # setup additional/legacy environment variables
    export CUSTOM_LLVM_ROOT=${ASTL_LLVM_DIR}
    export LLVM_VERSION=${ASTL_LLVM_VERSION}

    # pickup runtime environment that is to be built
    BUILD_DIR=$(git_root)/build
    export LD_LIBRARY_PATH=${BUILD_DIR}/lib:${LD_LIBRARY_PATH}
    export BUILD_DIR
  else
    echo "ERROR: Cannot determine LLVM-version!"
  fi
else
  echo "ERROR: Please source the ASTL-environment first!"
fi
