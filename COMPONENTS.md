# cuGraph Component Map

This repository keeps RAPIDS-style source paths, but the code is grouped into
four logical components.

## Native Core

Native core is the graph runtime substrate used by both algorithms and
sampling. It should not contain user-facing graph analytics semantics unless
the code is needed to build, store, move, or validate graph data.

Main paths:

- `cpp/src/detail`
- `cpp/src/structure`
- `cpp/src/utilities`
- `cpp/src/converters`
- `cpp/src/lookup`
- `cpp/src/mtmg`
- `cpp/include/cugraph`
- core C API files in `cpp/src/c_api`

Typical responsibilities:

- graph storage and graph views
- renumbering, relabeling, edge list conversion, transpose, symmetrization
- distributed shuffle and communication helpers
- validation and utility wrappers
- resource handles, arrays, errors, graph handles, and C API helper types

## Native Algorithms

Native algorithms are graph analytics and graph optimization kernels built on
top of core.

Main paths:

- `cpp/src/centrality`
- `cpp/src/community`
- `cpp/src/components`
- `cpp/src/cores`
- `cpp/src/dag`
- `cpp/src/generators`
- `cpp/src/layout`
- `cpp/src/linear_assignment`
- `cpp/src/link_analysis`
- `cpp/src/link_prediction`
- `cpp/src/traversal`
- `cpp/src/tree`
- algorithm C API files in `cpp/src/c_api`

Notes:

- `linear_assignment` means linear assignment / assignment matching. The
  current implementation includes the Hungarian algorithm for minimum-cost
  one-to-one assignment.
- `generators` are grouped with algorithms because they expose graph generation
  behavior rather than graph storage internals.

## Native Sampling

Native sampling is the graph sampling runtime in `libcugraph` and
`pylibcugraph`. It is used by GNN consumers, but it is not itself the PyG/GNN
integration layer.

Main paths:

- `cpp/src/sampling`
- `cpp/include/cugraph_c/sampling_algorithms.h`
- sampling C API files in `cpp/src/c_api`
- `python/pylibcugraph/src/pylibcugraph/sampling`
- `python/pylibcugraph/src/pylibcugraph/negative_sampling.py`
- `python/pylibcugraph/src/pylibcugraph/_bindings/sampling.cpp`
- `python/pylibcugraph/src/pylibcugraph/_bindings/negative_sampling.cpp`

Typical responsibilities:

- neighbor sampling
- temporal neighbor sampling
- biased sampling
- negative sampling
- random walks
- sampling result materialization

GNN-specific stores, loaders, and PyG classes belong in the sibling
`cugraph-gnn` repository.

## High-Level Python Algorithms

High-level Python algorithms are the public `cugraph` package layered over
`pylibcugraph`.

Main paths:

- `python/cugraph/src/cugraph`
- `python/cugraph/src/cugraph/dask`
- `python/cugraph/tests`

Typical responsibilities:

- Python graph classes and user-facing graph analytics APIs
- dask wrappers
- dataset helpers
- high-level tests and compatibility behavior

## CMake Classification

The CMake source group files under `cpp/cmake/sources` document and expose the
same logical split without moving source files:

- `core.cmake`
- `algorithms.cmake`
- `sampling.cmake`
- `c_api_core.cmake`
- `c_api_algorithms.cmake`
- `c_api_sampling.cmake`

These files currently classify sources for readability. They do not yet enable
or disable build profiles. Future build profile options should reuse these
component boundaries.
