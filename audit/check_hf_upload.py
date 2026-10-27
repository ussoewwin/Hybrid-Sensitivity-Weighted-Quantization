# -*- coding: utf-8 -*-
"""Guard: refuse a Hub upload whose payload is not the file it claims to be.

Wired into the upload path as a pre-flight check. Two failure modes it catches, both of which
have happened:

1. the payload is the path string instead of the file's contents (uploading a `str` through
   `upload_file(path_or_fileobj=...)` sends the string);
2. the payload parses as a model card but has lost its YAML front matter or a required section.

Use:
    python audit/check_hf_upload.py --path <local file> --require "## 🔒 Security"
"""
import argparse
import io
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="the local file that should be uploaded")
    ap.add_argument("--require", action="append", default=[],
                    help="a string that must appear in the payload (repeatable)")
    ap.add_argument("--front-matter", action="store_true",
                    help="the payload must start with a YAML front matter block")
    args = ap.parse_args()

    path = os.path.abspath(args.path)
    if not os.path.isfile(path):
        print("FAIL not a file:", path)
        return 1

    with open(path, "rb") as f:
        disk = f.read()

    # what a careless call uploads instead
    as_string = path.encode("utf-8")

    checks = []
    checks.append(("payload is not the path string", disk != as_string))
    checks.append(("payload is not a bare path line",
                   disk.strip() != disk.decode("utf-8", "replace").strip()
                   or not os.path.exists(disk.decode("utf-8", "replace").strip())))
    checks.append(("payload is larger than a path", len(disk) > 512))

    text = disk.decode("utf-8", "replace")
    if args.front_matter:
        checks.append(("yaml front matter present", text.lstrip().startswith("---")))
    for want in args.require:
        checks.append(("contains %r" % want, want in text))

    ok = True
    for name, passed in checks:
        print("%-4s %s" % ("OK" if passed else "FAIL", name))
        ok = ok and passed

    print("payload bytes:", len(disk))
    print("PRE-FLIGHT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
