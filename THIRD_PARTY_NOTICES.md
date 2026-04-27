# Third-party notices

This repository vendors snapshots of two external compressors so that the
compression-benchmarks experiment is reproducible from the supplementary zip
alone (no network access, no submodule resolution). Each row below
identifies the vendored snapshot; the corresponding sub-package contains the
upstream `LICENSE` (verbatim), a `README.md` with vendoring metadata, and a
`MODIFICATIONS.md` listing every edit applied on top.

| Sub-package           | Component | Upstream URL | Commit | License | Date vendored |
|-----------------------|-----------|--------------|--------|---------|---------------|
| `compressors/edpc/`   | EDPC      | _TBD_        | _TBD_  | _TBD_   | _TBD_         |
| `compressors/trace/`  | TRACE     | _TBD_        | _TBD_  | _TBD_   | _TBD_         |

Status: both rows are placeholders. Fill them in when the corresponding
upstream code is actually copied into `compressors/<name>/upstream/`. See
`compressors/<name>/upstream/README.md` for the per-codec metadata and the
project's refactor plan for the full vendoring procedure.

`compressors/transformer/` is our own work and has no `upstream/` directory.
