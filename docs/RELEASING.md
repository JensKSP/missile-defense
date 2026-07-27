# Releasing

Push a tag; a draft release appears with a build for every platform attached.
Everything below is what that sentence leaves out.

## Versioning

**Releases are SemVer, tagged `vX.Y.Z`.** Below 1.0 the *minor* is the breaking
lever, so `0.2.0` may break what `0.1.0` promised and `0.1.1` may not.

What counts as breaking here is worth stating, because the compatibility surface
is not an ABI. Three things invalidate work someone else has done:

| Surface | Why a change breaks it |
|---|---|
| Observation layout / action space | every trained policy becomes garbage |
| `.mdr` replay format | existing recordings refuse to load |
| Simulation behaviour | the scripted baseline's score stops being comparable |

A change to any of those is a minor bump before 1.0 and a major one after.

**The tree always carries the version it is working toward.** So a build off
`master` is a *pre-release* of that version, not a successor to the last one, and
`poe version` enforces that all three files agree at all times.

**One dev version, spelled the way each ecosystem sorts.** Derived from `git
describe` by [tools/version.py](../tools/version.py), never written by hand:

| Target | Format | Example |
|---|---|---|
| Git tag | `vX.Y.Z` | `v0.1.0` |
| SemVer / display | `X.Y.Z-dev.N+gSHA` | `0.1.0-dev.128+g83a3fe03` |
| Python (PEP 440) | `X.Y.Z.devN+gSHA` | `0.1.0.dev128+g83a3fe03` |
| Debian | `X.Y.Z~devN+gSHA` | `0.1.0~dev128+g83a3fe03` |
| Filenames | `X.Y.Z-devN-gSHA` | `0.1.0-dev128-g83a3fe03` |

The spellings are not decoration. Debian's `~` is the one character that sorts
before the empty string, and PEP 440's `.devN` does the same for pip — so a
nightly is *older* than the release it precedes. Get that backwards and an `apt
upgrade` pins a user on a build nobody has played. `N` counts commits since the
last tag, so nightlies also order among themselves. Print any of them with
`poe version --dev debian`.

## Cutting one

```bash
poe bump 0.2.0           # edit the three files, show the diff, stop
poe bump 0.2.0 --push    # ...commit, tag, and push — this starts the build
```

`--commit` and `--tag` are the intermediate steps; each flag implies the ones
before it, and the bare form is a dry run you can read first. It refuses to start
on a dirty tree, because a bump commit that quietly carries unrelated work is how
a release ends up shipping something nobody meant to include. Doing it by hand is
the same three files plus `git tag -a v0.2.0`.

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
3. **collects them**, renaming each distro's `.deb`s so the distribution is in
   the filename, and writes `SHA256SUMS`;
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
| **Ubuntu 26.04 LTS** | `*-ubuntu-26.04.deb` (game, bindings, console) | `ubuntu:26.04` |
| Debian trixie | `*-debian-trixie.deb` (game, bindings, console) | `debian:trixie` |
| Ubuntu 24.04 LTS (compatibility) | `missile-defense_<ver>-1_amd64-ubuntu-24.04.deb` | `ubuntu:24.04` |

Plus `SHA256SUMS` over all of them.

**The distro `.deb`s are not interchangeable.** They arrive from
`dpkg-buildpackage` with identical filenames and resolve against different Qt
and libstdc++ versions: Qt 6.10 on Ubuntu 26.04, 6.8 on trixie, and 6.4 on
Ubuntu 24.04. Hence the distribution in the filename: mixing them up is not a
subtle degradation, it is an install that fails at the far end.

Ubuntu 26.04 is the primary Ubuntu target and builds the game, Python bindings,
and training console. The 24.04 compatibility leg builds the game only because
that release has no archive package for nanobind.

## Nightlies

[nightly.yml](../.github/workflows/nightly.yml) rebuilds `master` at 03:15 UTC
and replaces a pre-release on the rolling `nightly` tag, so
[one URL](https://github.com/JensKSP/missile-defense/releases/tag/nightly) always
has the newest build and the README can link it.

It calls the same [build.yml](../.github/workflows/build.yml) as everything else,
passing the dev version so the filenames say what they are — a file called
`missile-defense-0.1.0.dmg` that is not 0.1.0 is worse than useless once it is in
someone's Downloads folder.

Two details worth knowing:

- **It skips when `master` has not moved.** The previous nightly's target commit
  is compared against `HEAD` first. Rebuilding an identical tree costs four
  platforms of runner time and churns the download URLs for nothing.
- **Marked pre-release**, so it never becomes "Latest release" and
  `/releases/latest` keeps pointing at a version someone has looked at.

Workflow artifacts cannot do this job: they expire after 90 days, they can only
be downloaded by someone signed in to GitHub, and their URLs are per-run. A
release asset has none of those limits.

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
