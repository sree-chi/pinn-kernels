# GPU image (default):  docker build -t pinn-kernels:gpu .
#                       docker run --gpus all --rm pinn-kernels:gpu make bench-gpu
# CPU image:            docker build --build-arg BASE=ubuntu:24.04 \
#                         --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t pinn-kernels:cpu .
#                       docker run --rm pinn-kernels:cpu
ARG BASE=nvidia/cuda:12.6.3-devel-ubuntu24.04
FROM ${BASE}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Ubuntu 24.04 ships LLVM 18 (the version the code generator targets).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential cmake git \
        llvm-18-dev libzstd-dev libedit-dev zlib1g-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv $VIRTUAL_ENV

WORKDIR /app

# Dependencies first for layer caching.
RUN pip install --index-url ${TORCH_INDEX} torch \
    && pip install numpy matplotlib pytest \
    && if [ -n "$(echo ${TORCH_INDEX} | grep -v /cpu)" ]; then pip install triton cupy-cuda12x; fi

COPY csrc csrc
COPY pinn_kernels pinn_kernels
COPY pyproject.toml Makefile README.md ./
RUN make build && pip install -e .
COPY tests tests

CMD ["sh", "-c", "python -m pytest -q && python -m pinn_kernels.bench kernels"]
