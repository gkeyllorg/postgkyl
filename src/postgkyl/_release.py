"""Git-derived release numbers; standard library only for build-time use."""

from pathlib import Path
import re
import subprocess

BASE_VERSION = (2, 0, 0, 0)
BASE_COMMIT = "f45fb37ce4ca9e0a93ab8c19e960264f77f93cb2"
VERSION_FILE = "_release_version.txt"
SOURCE_PATH = "src/postgkyl/_release.py"


def git(root: Path, *args: str) -> str:
  return subprocess.run(["git", "-C", str(root), *args],
                        check=True,
                        capture_output=True,
                        text=True).stdout.strip()


def release_anchor(root: Path) -> tuple[tuple[int, ...], str, list[str]]:
  """Return the latest first-parent release and subsequent commits, oldest first.

  Only four-part vMAJOR.MINOR.PR.0 tags are release markers. When several
  releases share a commit, the greatest version wins. Side-branch tags do
  not change a branch's version.
  """
  if git(root, "rev-parse", "--is-shallow-repository") == "true":
    raise RuntimeError(
        "Versioning requires full Git history: git fetch --unshallow --tags")
  tags = {}
  refs = git(root, "for-each-ref",
             "--format=%(refname:short) %(objectname) %(*objectname)",
             "refs/tags/v*")
  for ref in refs.splitlines():
    tag, *objects = ref.split()
    if re.fullmatch(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\.0", tag):
      number = tuple(map(int, tag[1:].split(".")))
      if number >= BASE_VERSION:
        commit = objects[-1]  # peeled commit for annotated tags
        tags[commit] = max(tags.get(commit, BASE_VERSION), number)
  history = git(root, "rev-list", "--first-parent", "HEAD").splitlines()
  for distance, commit in enumerate(history):
    if commit in tags or commit == BASE_COMMIT:
      return tags.get(commit, BASE_VERSION), commit, history[:distance][::-1]
  # Squash/rebase merging the initial feature branch can remove BASE_COMMIT
  # from main's ancestry. Start immediately before versioning was introduced
  # so the PR that introduces it is itself counted by release automation.
  introduced = git(root, "log", "--first-parent", "--diff-filter=A",
                   "--format=%H", "--", SOURCE_PATH).splitlines()
  if not introduced or introduced[-1] == history[-1]:
    raise RuntimeError("Cannot locate the Postgkyl version baseline")
  index = history.index(introduced[-1])
  anchor = history[index + 1]
  return BASE_VERSION, anchor, history[:index + 1][::-1]


def get_version(package_dir: Path) -> str:
  """Use immutable artifact metadata, or derive a source checkout's version."""
  baked = package_dir / VERSION_FILE
  if baked.is_file():
    return baked.read_text().strip()
  root = package_dir.parents[1]
  if not (root / ".git").exists():
    raise RuntimeError(
        "Install Postgkyl from a Git clone or a built sdist/wheel")
  number, _, commits = release_anchor(root)
  return ".".join(map(str, (*number[:3], len(commits))))


__version__ = get_version(Path(__file__).resolve().parent)
