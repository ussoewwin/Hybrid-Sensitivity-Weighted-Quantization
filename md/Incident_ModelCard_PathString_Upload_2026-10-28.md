# Incident - a model card overwritten with a local path string (2026-10-28)

## What happened

The SAM 3.1 ConvRot INT8 model card on the Hub
(`ussoewwin/SAM3.1-HSWQ-ConvRot-INT8`) was replaced with the literal text of a local file path:

```
D:\\USERFILES\\GitHub\\hswq\\test\\_hf_README_sam3.1_convrot_int8.md
```

63 bytes replaced a 5,081-byte card, including its YAML front matter. The card was public in
that state until the next commit restored it.

## Cause

`huggingface_hub.HfApi.upload_file` was called with `path_or_fileobj=<a path string>`. That
argument accepts **either** a path **or** bytes/file-like content; given a `str`, it uploads
the string itself as the file body. The call looked correct because the same variable held the
path used for reading, and the reading step (`io.open(...).read()`) was in a different
statement.

The upload reported success. The library only warned about missing YAML metadata in the body -
which was the warning that should have been read as "the body is not a model card".

Two checks were skipped that would each have caught it before the commit:

1. **no verification of the content that was about to be uploaded** (bytes, front matter
   present, expected section present) - the diff check compared the *local* files, not the
   payload;
2. **no read-back immediately after the upload** - the read-back ran, but its result
   (`bytes 63`, `security section live: False`, `matches local: False`) was printed and not
   treated as a failure, so a known-broken state was left in place until the next step.

## Repair

Restored by uploading the file **by path** (`path_or_fileobj=<the .md path>`, which the library
reads) from the repository copy `test/_hf_README_sam3.1_convrot_int8.md`, which is the same
document the card held before the change plus the new Security section. Verified: live card
6,322 bytes, YAML front matter present, `## 🔒 Security` present, byte-identical to the local
file.

Blast radius checked: all 19 repositories under the account were read back; no other card was
affected (two minimal cards that predate this work are unrelated and unchanged).

## Rules that follow

1. **Upload the file by path, never a path string.** Pass a file path or an open file object,
   not a variable that may hold a string of the path. If bytes are constructed, build them from
   the read result in the same statement.
2. **Verify the payload before it leaves.** For a model card: byte count equal to the local
   file, front matter starting with `---`, the new section present. Refuse to upload otherwise.
3. **Read back and gate on the result.** After any Hub write, fetch the live file and compare it
   byte for byte with the local file. A mismatch is a failure that must be repaired in the same
   turn - printing the mismatch is not a report, it is an unresolved break.
4. **Treat library warnings about the payload as errors.** "empty or missing yaml metadata in
   repo card" on a card upload means the body is not a card.
5. **A public card is a public surface.** The check is not "did the call succeed" but "what does
   a reader see now", verified by reading it.

## Record

Live card: `ussoewwin/SAM3.1-HSWQ-ConvRot-INT8/README.md`, 6,322 bytes, restored at the commit
after the breaking one. Local source of truth: `test/_hf_README_sam3.1_convrot_int8.md`
(repository, commit `7301328`, the Security section restored in the following commit).
