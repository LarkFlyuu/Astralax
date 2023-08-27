#!/usr/bin/env bash
#
# Script for Buildkite automation only.
# Environment variables must have been declared already.
#
# Check LLVM installation and trigger the build if not.

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

ASTLROOT=$(realpath ${ASTLROOT})
if [ ! -f ${ASTLROOT}/enable ]; then
  echo "'${OPTARG}' not a ASTLROOT directory"
  exit 1
fi

LLVMROOT=$(realpath ${LLVMROOT})
if [ ! -d ${LLVMROOT} ]; then
  echo "'${OPTARG}' not a LLVMROOT directory"
  exit 1
fi

# Find LLVM_VERSION
LLVM_VERSION=$(llvm_version)

# If not found, trigger a build
if [ ! -d "${LLVMROOT}/${LLVM_VERSION}" ]; then
  ${ASTLROOT}/trigger.sh astl-llvm ${LLVM_VERSION}
else
  echo "Found $LLVM_VERSION"
fi
