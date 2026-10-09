# SGLang CP-sharded MoE on A100

Personal research project based on [SGLang](https://github.com/sgl-project/sglang), focused on distributed long-context MoE inference on 4–8 NVIDIA A100 SXM 80 GB GPUs.

## Goal

Implement and evaluate a route-aware MoE dispatcher for context-parallel prefill. The planned path keeps tokens on their CP owners, sends hidden states only to ranks that own selected experts, and returns combined outputs to the original token order. The existing SGLang path will serve as the correctness and performance baseline.

Current status: upstream baseline imported; the new dispatcher and performance results are pending.

## Repository layout

- `python/`: SGLang runtime and MoE implementation.
- `benchmark/`: performance workloads and benchmarks.
- `test/`: correctness and regression tests.
- `scripts/`, `docker/`, `3rdparty/`, `proto/`, and `rust/`: supporting build and development components retained from SGLang.

## Provenance and license

Based on SGLang commit `00bcc25f6c857dc7cfb9e624ee38269d4c3735f6`. SGLang and its contributors retain their original notices and Apache-2.0 license; see [LICENSE](LICENSE). Third-party license files remain with their respective components. New project changes will be documented separately.
