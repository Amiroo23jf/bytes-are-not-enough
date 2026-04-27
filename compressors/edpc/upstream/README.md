# EDPC (vendored)

Status: NOT YET VENDORED.

When the upstream EDPC repository is vendored into this directory, fill in:

- Source URL: <upstream URL>
- Commit hash: <40-char SHA>
- Date vendored: <YYYY-MM-DD>
- License: <e.g. MIT, Apache-2.0> (preserved verbatim in `LICENSE`)
- Files removed during cleanup: <e.g. examples/, notebooks/, paper/, .github/>
- Vendoring command (for reproducibility): <git clone + cp commands>

Files imported by `compressors/edpc/adapter.py` should live directly under
this directory (no `src/`, `python/` indirection) so that imports stay
explicit:

```python
from compressors.edpc.upstream.<module> import <Class>
```

See `MODIFICATIONS.md` for the running list of edits applied on top of the
original upstream snapshot.
