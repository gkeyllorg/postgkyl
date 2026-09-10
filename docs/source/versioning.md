# Version numbers

Postgkyl uses `major.minor.PR.commit`, starting at `2.0.0.0` for commit
`f45fb37ce4ca9e0a93ab8c19e960264f77f93cb2`.

| Event | Example version |
| --- | --- |
| Baseline | `2.0.0.0` |
| Two subsequent commits | `2.0.0.2` |
| First merged PR | `2.0.1.0` |
| One subsequent commit | `2.0.1.1` |
| Next merged PR | `2.0.2.0` |
| Explicit minor release | `2.1.0.0` |
| Explicit major release | `3.0.0.0` |

`PR` is a sequential count of merged pull requests within the current
major/minor release, not the GitHub issue/PR identifier. `commit` counts
first-parent commits since the release tag: a merge commit counts once,
regardless of the number of commits on its side branch. Uncommitted changes
do not increment the version; `pgkyl --version` separately reports dirty state.

## Automatic releases

Once these changes reach `main`, the **Version** GitHub Actions workflow
reconciles merged PRs on every push. It creates annotated tags such as
`v2.0.1.0` on the PR's final merge commit. It supports merge commits, squash
merges, and rebase merges, using GitHub's
[associated pull requests API](https://docs.github.com/en/rest/commits/commits#list-pull-requests-associated-with-a-commit).
Direct pushes increment only the commit count.

Each run catches up on all merges since the last release, in history order.
Reruns do not count a merge twice. Tags are pushed atomically, and the main
CI build runs after successful version maintenance so its artifacts use the
new tags. No source-changing bot commits are needed. This workflow maintains
version tags; it does not publish packages to PyPI.

If the initial PR is squashed or rebased so the original baseline is absent
from main's first-parent history, automation starts immediately before the
commit introducing version support and counts that PR as the first release.

For an explicit minor or major release, run **Actions → Version → Run
workflow** on `main` and select `minor` or `major`. The workflow first catches
up on merged PRs, then resets the lower components. The workflow needs its
declared repository `contents: write` and `pull-requests: read` permissions;
repository tag rules must allow it to create `v*` tags.

## Local installs and distribution artifacts

Use a full Git clone and fetch release tags with `git fetch --tags`. A shallow
checkout must first run `git fetch --unshallow --tags`; it fails explicitly
instead of reporting an incorrect commit count. Linked worktrees are supported.

Source and editable imports compute the version from the current checkout.
Restart a running Python process after changing commits or fetching tags.
Installed package metadata is a build/install snapshot, so reinstall an editable
package when you need that metadata to reflect a new checkout version.

Wheels and source distributions bake in the same version as their package
metadata. They work without Git, and rebuilding a wheel from an sdist preserves
the version. Plain GitHub source ZIPs lack the required history and baked
metadata; use a clone or a built distribution instead.
