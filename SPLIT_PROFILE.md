# cuGraph Repository Profile

This repository is the graph analytics half of the local two-repo split.

## Ownership

- Native cuGraph runtime and C API.
- Graph algorithms and sampling primitives used by cuGraph consumers.
- `libcugraph`, `pylibcugraph`, and high-level `cugraph` packages.

## Non-Goals

- PyG loaders and samplers.
- WholeGraph native runtime.
- WholeGraph PyTorch adapter.

Those pieces belong in the sibling `cugraph-gnn` repository.

## Layout Decision

The tree intentionally stays close to RAPIDS cuGraph conventions:

- Native source remains under `cpp/include`, `cpp/src`, and `cpp/tests`.
- Python packages remain under `python/<package>`.
- C API headers stay under `cpp/include/cugraph_c`.
- C API implementations stay under `cpp/src/c_api`.

This avoids adding `cpp/core`, `cpp/algorithms`, and `cpp/sampling` wrapper
directories before the build graph actually needs that extra nesting.
