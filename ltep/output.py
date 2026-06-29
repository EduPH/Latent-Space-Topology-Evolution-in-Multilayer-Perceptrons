"""
ltep.output -- per-run output folders.

Every experiment writes all of its artefacts into one folder

    <root>/<dataset>/<tag>/
        log.txt        full stdout+stderr of the run (mirrored, console still shows it)
        params.json    the exact parameters the run used
        *.png          all figures

`<root>` defaults to ./results (override with the LTEP_RESULTS env var). `<tag>`
is the dataset's run signature (e.g. COIL's gs2_ss0.80_...); a timestamp is
appended so repeated runs don't overwrite each other.

Usage in an experiment's __main__:

    from ltep import output
    rd = output.run_dir("cardio", tag="alpha0.01")
    output.save_params(rd, PARAMS)
    with output.capture(rd):            # mirror stdout/stderr -> rd/log.txt
        main(outdir=rd, ...)
"""
import os
import sys
import json
import datetime

DEFAULT_ROOT = os.environ.get("LTEP_RESULTS", "results")


def run_dir(dataset, tag=None, root=None, timestamp=True):
    """Create and return <root>/<dataset>/<tag>[_<timestamp>]/ (absolute path).
    Absolute so it keeps working even if the process later changes directory."""
    root = root or DEFAULT_ROOT
    parts = [tag] if tag else []
    if timestamp:
        parts.append(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    sub = "_".join(parts) if parts else "run"
    path = os.path.abspath(os.path.join(root, dataset, sub))
    os.makedirs(path, exist_ok=True)
    return path


def save_params(out_dir, params, name="params.json"):
    """Dump the run parameters next to the figures, for provenance."""
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(dict(params), f, indent=2, default=str)
    return os.path.join(out_dir, name)


class _Fan:
    """Write to several streams at once (console + logfile)."""
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


class capture:
    """Context manager: mirror stdout AND stderr into <out_dir>/<name> while still
    printing to the console (like `tee`). Note: C-level writes (e.g. some TensorFlow
    absl banners) bypass Python streams and may not be captured -- the pipeline's own
    prints all are."""
    def __init__(self, out_dir, name="log.txt"):
        self.path = os.path.join(out_dir, name)
        self._f = self._out = self._err = None

    def __enter__(self):
        self._f = open(self.path, "w", buffering=1)            # line-buffered
        self._out, self._err = sys.stdout, sys.stderr
        sys.stdout = _Fan(self._out, self._f)
        sys.stderr = _Fan(self._err, self._f)
        return self.path

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self._out, self._err
        if self._f:
            self._f.close()
        return False
