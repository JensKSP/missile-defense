# Releasing

Push a tag; a draft release appears with a build for every platform attached.
Everything below is what that sentence leaves out.

## Cutting one

```bash
# 1 — bump the version in all three places that carry it
#     CMakeLists.txt (project VERSION), pyproject.toml, debian/changelog
poe version v0.2.0          # fails unless the three agree, and agree with the tag

# 2 — tag and push
git tag -a v0.2.0 -m "v0.2.0"
git push origin v0.2.0
```

[.github/workflows/release.yml](../.github/workflows/release.yml) then:

1. **checks the tag against the tree** before spending twenty minutes of runner
   time. A tag that disagrees with `CMakeLists.txt` would build artifacts named
   after the tree and publish them under a release named after the tag, and
   nothing further down would notice;
2. **builds every artifact** by calling
   [build.yml](../.github/workflows/build.yml) — the *same* file
   [ci.yml](../.github/workflows/ci.yml) calls on every push, so a release ships
   what CI has been building all along rather than the output of a second recipe
   that nobody runs between releases;
3. **collects them**, renaming the two `.deb`s so the distribution is in the
   filename, and writes `SHA256SUMS`;
4. **opens a draft release** with the download guide above GitHub's generated
   commit notes.

## Why it stops at a draft

Everything before the last step is reproducible from the tag. Publishing is not:
it mails watchers and hands the world a download. So the artifacts are attached
and checkable first, and the release goes out when you press the button — after
downloading at least one of them and confirming it runs.

To publish automatically instead, drop `--draft` from the last step.

## What comes out

| Platform | Asset | Built on |
|---|---|---|
| macOS (Apple silicon) | `missile-defense-<ver>-macos-arm64.dmg` | `macos-15` |
| Windows | `missile-defense-<ver>-win64.exe`, `…-win64.zip` | MSYS2 CLANG64 |
| Debian trixie | `missile-defense_<ver>-1_amd64-debian-trixie.deb` | `debian:trixie` |
| Ubuntu 24.04 | `missile-defense_<ver>-1_amd64-ubuntu-24.04.deb` | `ubuntu:24.04` |

Plus `SHA256SUMS` over all of them.

**The two `.deb`s are not interchangeable.** They arrive from
`dpkg-buildpackage` with byte-identical filenames and resolve against different
Qt ABIs — `qt6-base-private-abi (= 6.8.2)` on trixie against
`qt6-base-abi (= 6.4.2)` on Ubuntu, with `libqt6gui6` renamed `libqt6gui6t64` in
between. Hence the distribution in the filename: mixing them up is not a subtle
degradation, it is an install that fails at the far end.

## Rehearsing it

`workflow_dispatch` runs the whole path against an existing tag — including the
draft — so the mechanism can be exercised without spending a new tag on it. A tag
is public and permanent; a dry run should be neither.

Delete a draft release and its tag with:

```bash
gh release delete v0.2.0 --yes
git push origin :refs/tags/v0.2.0 && git tag -d v0.2.0
```

## Known limits

- **The macOS bundle is ad-hoc signed, not notarised.** Gatekeeper refuses it
  until the user clears quarantine.
  [MACOS.md](MACOS.md#signing-it-for-other-people) has the Developer ID path,
  which needs a paid Apple account and nothing else.
- **Nobody has run the macOS build.** It compiles, all 104 tests pass on Apple
  silicon, the `.dmg` is well-formed — and no human has seen a frame of it.
- **No source tarball.** `debian/source/format` is `3.0 (quilt)`, which wants an
  orig tarball a git checkout does not have, so the packages are built
  binary-only. GitHub's own auto-generated source archives are attached to every
  release regardless.
- **Single architecture per platform.** The `.dmg` is arm64 only, because
  Homebrew's Qt is not universal; the `.deb`s are amd64.
