# cuGraph

This repository owns the RAPIDS cuGraph graph analytics stack.

It keeps the native cuGraph runtime, the Python bindings, and the high-level
`cugraph` package in one RAPIDS-style source tree. GNN integration and
WholeGraph live in the sibling `cugraph-gnn` repository.

## Packages

- `libcugraph`: C++/CUDA graph runtime, C API, algorithms, and sampling
  primitives.
- `pylibcugraph`: Python bindings for native cuGraph graph construction,
  algorithms, communication, resource helpers, and sampling.
- `cugraph`: high-level Python graph analytics package.

## Layout

- `cpp`: native cuGraph sources, public headers, tests, docs, and examples.
- `cpp/libcugraph_etl`: native ETL helper library.
- `python/libcugraph`: wheel wrapper for the native runtime.
- `python/pylibcugraph`: Python bindings and tests.
- `python/cugraph`: high-level Python package.
- `conda/recipes`: package recipes for the owned distributions.

See `COMPONENTS.md` for the logical core, algorithms, sampling, and
high-level Python boundaries inside this RAPIDS-style layout.

## Build

The no-argument build follows the local dependency order:

```bash
./build.sh
```

Equivalent explicit targets:

```bash
./build.sh libcugraph pylibcugraph cugraph
```
