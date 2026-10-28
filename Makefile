PY ?= python3
LLVM_DIR ?= /usr/lib/llvm-18/lib/cmake/llvm

.PHONY: build install test bench-cpu bench-gpu train clean docker-cpu docker-gpu

build:
	cmake -S csrc -B build -DLLVM_DIR=$(LLVM_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j

install: build
	$(PY) -m pip install -e ".[dev]"

test: build
	$(PY) -m pytest -q

bench-cpu: build
	$(PY) -m pinn_kernels.bench kernels --device cpu
	$(PY) -m pinn_kernels.bench train --device cpu

bench-gpu: build
	$(PY) -m pinn_kernels.bench kernels --device cuda
	$(PY) -m pinn_kernels.bench train --device cuda

train: build
	$(PY) -m pinn_kernels.train --problem inverse --physics fd --backend llvm --out results/inverse.json --plot results/inverse.png

docker-cpu:
	docker build --build-arg BASE=ubuntu:24.04 --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t pinn-kernels:cpu .

docker-gpu:
	docker build -t pinn-kernels:gpu .

clean:
	rm -rf build pinn_kernels/_lib .pytest_cache
