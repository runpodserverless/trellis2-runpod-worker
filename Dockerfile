# TRELLIS.2 — RunPod Serverless worker
#
# Requirements (from the official repo, github.com/microsoft/TRELLIS.2):
#   - CUDA 12.4
#   - NVIDIA GPU with >= 24GB VRAM (validated on A100/H100; RTX 4090 works fine)
#   - Linux only
#
# Design choices:
#   - Base image is nvidia/cuda:12.4.0-devel so the CUDA toolkit needed to
#     compile flash-attn/nvdiffrast/cumesh (native extensions) is present.
#   - We install into a conda env named "trellis2", matching the official
#     setup.sh so any troubleshooting advice from the upstream repo still
#     applies verbatim.
#   - Model weights are downloaded at BUILD time (not at container start).
#     RunPod Serverless spins up fresh containers on cold start; downloading
#     ~8GB of weights on every cold start would make first-request latency
#     unacceptable. Baking them into the image trades a larger image (slower
#     initial push/pull, one time) for much faster cold starts (every request
#     after that).

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Docker's default RUN shell is "/bin/sh -c", which on Ubuntu is dash, not
# bash. The upstream setup.sh is written for bash. Sourcing it under dash
# (the original bug here) can silently swallow a failed `conda create`
# instead of failing the build — the env never gets created, and the next
# RUN step dies later with a confusing "exit code 127". Forcing bash for
# every RUN in this file removes that whole class of failure.
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PATH=/opt/conda/bin:$PATH

# --- System dependencies ---------------------------------------------------
RUN apt-get update && apt-get install -y \
    git wget curl build-essential ninja-build \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# --- Miniconda (the repo recommends conda for dependency management) -------
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

WORKDIR /workspace

# --- Clone TRELLIS.2 with submodules ----------------------------------------
# --recursive is not optional: the repo uses git submodules for some of its
# rendering dependencies. Forgetting it produces cryptic import errors much
# later in the build, not a clear "submodule missing" message.
RUN git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
WORKDIR /workspace/TRELLIS.2

# --- Install dependencies via the official setup script ---------------------
# Flags match the ones documented in the repo's README:
#   --new-env      : create the "trellis2" conda env
#   --basic        : core Python dependencies (torch 2.6.0 + CUDA 12.4, etc.)
#   --flash-attn   : attention backend (fast path; skip + use xformers on
#                    GPUs that don't support it, e.g. V100 — not relevant for
#                    RunPod's A100/H100/RTX 4090 offerings)
#   --nvdiffrast, --nvdiffrec, --cumesh, --o-voxel, --flexgemm :
#                    rendering / mesh-processing extensions the pipeline
#                    needs for texture baking and mesh export
# This step is slow (native extensions compiled from source) — expect
# 20-40 minutes on a typical CI runner.
#
# GitHub-hosted runners (ubuntu-latest) have no NVIDIA driver at all, so
# setup.sh's own platform check — `command -v nvidia-smi` — fails instantly
# and the script exits before installing anything (this is what produced
# "Error: No supported GPU found"). That check only cares whether the
# *command* exists, not whether a GPU is attached, so a minimal stub is
# enough to get past it. The actual compilation (nvcc, flash-attn,
# nvdiffrast, cumesh) doesn't need a live device — the CUDA Toolkit is
# already in this base image — as long as we tell it explicitly which
# architectures to build for instead of letting it try to auto-detect one.
RUN printf '#!/bin/sh\ncase "$*" in\n  *--query-gpu*) echo "A100-SXM4-80GB, 81920" ;;\n  *) echo "Stub nvidia-smi (no physical GPU on this build host)" ;;\nesac\nexit 0\n' \
    > /usr/local/bin/nvidia-smi \
    && chmod +x /usr/local/bin/nvidia-smi

# A100=8.0, A6000/3090=8.6, RTX 4090=8.9, H100=9.0 — covers the GPUs named
# in the README. FORCE_CUDA=1 stops PyTorch's cpp_extension build tooling
# from skipping the CUDA build when torch.cuda.is_available() is False
# (which it always is here, since there's no device — only a stub).
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1

# Run with `bash` (not `. `/source, and not the default dash shell) so any
# internal failure actually propagates. The `test -x ... && conda env list`
# tail is a hard checkpoint: if setup.sh didn't actually create the
# "trellis2" env, the build fails right here with a clear message instead
# of a mystifying "exit code 127" two steps later on the pip install.
RUN bash ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm \
    && test -x /opt/conda/envs/trellis2/bin/pip \
    && conda env list

# --- RunPod SDK + our own worker dependencies -------------------------------
# Installed into the trellis2 env specifically, not the base Python, so the
# handler can import trellis2's own modules directly.
RUN /opt/conda/envs/trellis2/bin/pip install --no-cache-dir \
    runpod \
    pillow \
    requests \
    huggingface_hub

# --- Pre-download model weights at build time -------------------------------
# microsoft/TRELLIS.2-4B is a public model; no HF token required. This layer
# is the main reason the image is large, and the main reason cold starts
# stay fast: the alternative is an 8GB+ download on every fresh worker.
RUN /opt/conda/envs/trellis2/bin/python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='microsoft/TRELLIS.2-4B')"

# --- Worker code -------------------------------------------------------------
COPY handler.py /workspace/TRELLIS.2/handler.py
COPY test_input.json /workspace/TRELLIS.2/test_input.json

# RunPod's serverless harness expects the container's default command to
# start the handler loop directly — no shell prompt, no server framework.
ENTRYPOINT ["/opt/conda/envs/trellis2/bin/python3", "-u", "handler.py"]