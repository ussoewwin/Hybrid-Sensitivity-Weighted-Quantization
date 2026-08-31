"""ComfyUI comfy package marker.

Without this file the comfy directory is treated as a PEP 420 namespace
package, so Python prefers any regular (``__init__.py``-bearing) comfy package
found later on sys.path -- e.g. a pip-installed ``comfy`` in site-packages on
cloud hosts. That shadows this tree and breaks repo-local shims such as
``comfy/options.py``. Marking it as a regular package keeps imports pinned to
the ComfyUI tree this repo ships.
"""
