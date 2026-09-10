"""Reconcile merged PRs on main into release tags, then optionally bump a release."""

import argparse
import json
import os
from pathlib import Path
import runpy
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RELEASE = runpy.run_path(ROOT / "src/postgkyl/_release.py")
git = RELEASE["git"]
release_anchor = RELEASE["release_anchor"]


def is_merged_pr(repository: str, token: str, commit: str) -> bool:
  """Recognize merge, squash, and rebase PRs by GitHub's final merge SHA."""
  page = 1
  while True:
    request = Request(
        f"https://api.github.com/repos/{repository}/commits/{commit}/pulls"
        f"?per_page=100&page={page}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        })
    with urlopen(request, timeout=30) as response:
      pulls = json.load(response)
    if any(
        pr["merged_at"] and pr["base"]["ref"] == "main" and pr["base"]["repo"]
        ["full_name"] == repository and pr["merge_commit_sha"] == commit
        for pr in pulls):
      return True
    if len(pulls) < 100:
      return False
    page += 1


def tag_releases(root: Path,
                 merged_pr,
                 bump: str = "none",
                 run_id: str = "") -> list[str]:
  """Tag every unprocessed merge in history order, making retries harmless."""
  marker = f"Postgkyl release workflow: {run_id}"
  if run_id and marker in git(root, "for-each-ref", "--format=%(contents)",
                              "refs/tags/v*").splitlines():
    return []
  number, _, commits = release_anchor(root)
  tags = []
  for commit in commits:
    if merged_pr(commit):
      number = (*number[:2], number[2] + 1, 0)
      tag = "v" + ".".join(map(str, number))
      git(root, "tag", "-a", tag, commit, "-m", "Merged PR release")
      tags.append(tag)
  if bump != "none":
    number = ((number[0] + 1, 0, 0, 0) if bump == "major" else
              (number[0], number[1] + 1, 0, 0))
    tag = "v" + ".".join(map(str, number))
    git(root, "tag", "-a", tag, "HEAD", "-m", marker)
    tags.append(tag)
  return tags


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--bump",
                      choices=("none", "minor", "major"),
                      default="none")
  args = parser.parse_args()
  repository = os.environ["GITHUB_REPOSITORY"]
  token = os.environ["GH_TOKEN"]
  tags = tag_releases(ROOT, lambda sha: is_merged_pr(repository, token, sha),
                      args.bump, os.environ["GITHUB_RUN_ID"])
  if tags:
    # One atomic push: a failed push leaves no partially published release.
    git(ROOT, "push", "--atomic", "origin",
        *(f"refs/tags/{tag}" for tag in tags))
  print("Created " + ", ".join(tags) if tags else "Release tags are up to date")


if __name__ == "__main__":
  main()
