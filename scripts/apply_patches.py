#!/usr/bin/env python
"""Apply the runtime patches the full companion (moshi.reactive_server) needs, to your
installed/vendored moshi package. Idempotent — safe to run more than once.

Currently: makes LMGen.on_text_hook able to RETURN a replacement text token (the greeting-prime
forces the opening this way). Existing hooks that return None are unaffected.

Usage:  python scripts/apply_patches.py            # patches the moshi on PYTHONPATH
        python scripts/apply_patches.py /path/to/moshi   # or a specific package dir
"""
import os, sys, importlib.util

def find_lm_py():
    if len(sys.argv) > 1:
        base = sys.argv[1]
        cand = os.path.join(base, "models", "lm.py")
        if os.path.exists(cand):
            return cand
        cand = os.path.join(base, "lm.py")
        if os.path.exists(cand):
            return cand
    spec = importlib.util.find_spec("moshi")
    if spec and spec.submodule_search_locations:
        return os.path.join(list(spec.submodule_search_locations)[0], "models", "lm.py")
    raise SystemExit("could not locate moshi/models/lm.py; pass the moshi package dir as an argument")

def main():
    f = find_lm_py()
    s = open(f).read()
    if "_rep = self.on_text_hook" in s:
        print(f"already patched: {f}")
        return
    old = "        if self.on_text_hook is not None:\n            self.on_text_hook(text_token)\n"
    new = ("        if self.on_text_hook is not None:\n"
           "            _rep = self.on_text_hook(text_token)\n"
           "            if _rep is not None:\n"
           "                text_token = _rep\n")
    if old not in s:
        raise SystemExit(f"anchor not found in {f} — moshi version may differ; patch manually "
                         "(make on_text_hook's return value replace text_token).")
    open(f, "w").write(s.replace(old, new, 1))
    print(f"patched on_text_hook return in: {f}")

if __name__ == "__main__":
    main()
