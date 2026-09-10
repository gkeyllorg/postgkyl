"""Release numbering against real Git histories, without GitHub writes."""

import importlib
import io
import json
from pathlib import Path
import runpy
import subprocess

import pytest

release = importlib.import_module("postgkyl._release")
automation = runpy.run_path(
    Path(__file__).parents[1] / "scripts/tag_release.py")
pytestmark = pytest.mark.compatibility


@pytest.fixture
def repo(tmp_path):
  release.git(tmp_path, "init", "-b", "main")
  release.git(tmp_path, "config", "user.name", "Version tests")
  release.git(tmp_path, "config", "user.email", "version@example.com")
  release.git(tmp_path, "commit", "--allow-empty", "-m", "baseline")
  release.git(tmp_path, "tag", "v2.0.0.0")
  (tmp_path / "src/postgkyl").mkdir(parents=True)
  return tmp_path


def commit(repo, message="change"):
  release.git(repo, "commit", "--allow-empty", "-m", message)
  return release.git(repo, "rev-parse", "HEAD")


def version(repo):
  return release.get_version(repo / "src/postgkyl")


def test_current_baseline_without_a_tag(repo, monkeypatch):
  release.git(repo, "tag", "-d", "v2.0.0.0")
  monkeypatch.setattr(release, "BASE_COMMIT",
                      release.git(repo, "rev-parse", "HEAD"))
  assert version(repo) == "2.0.0.0"
  commit(repo)
  assert version(repo) == "2.0.0.1"


def test_commits_and_release_resets(repo):
  assert version(repo) == "2.0.0.0"
  commit(repo)
  commit(repo)
  assert version(repo) == "2.0.0.2"
  release.git(repo, "tag", "-a", "v2.0.1.0", "-m", "PR")
  assert version(repo) == "2.0.1.0"
  commit(repo)
  assert version(repo) == "2.0.1.1"
  release.git(repo, "tag", "v2.1.0.0")
  assert version(repo) == "2.1.0.0"
  release.git(repo, "tag", "v3.0.0.0")
  assert version(repo) == "3.0.0.0"


def test_side_branch_tags_and_commits_do_not_inflate_version(repo):
  release.git(repo, "checkout", "-b", "feature")
  commit(repo)
  commit(repo)
  release.git(repo, "tag", "v9.0.0.0")
  release.git(repo, "checkout", "main")
  release.git(repo, "merge", "--no-ff", "feature", "-m", "merge PR")
  assert version(repo) == "2.0.0.1"


def test_legacy_and_unrelated_tags_are_ignored(repo):
  commit(repo)
  for tag in ("v1.7.0", "v1.9.0.0", "v2.0.99.1", "v2.0.beta.0", "v02.0.9.0"):
    release.git(repo, "tag", tag)
  assert version(repo) == "2.0.0.1"


@pytest.mark.parametrize("method", ["squash", "merge", "rebase"])
def test_initial_merge_counts_the_introducing_pr(repo, method):
  release.git(repo, "tag", "-d", "v2.0.0.0")
  release.git(repo, "checkout", "-b", "feature")
  source = repo / release.SOURCE_PATH
  source.write_text("# version support introduced in this PR\n")
  release.git(repo, "add", release.SOURCE_PATH)
  commit(repo)
  commit(repo, "finish version automation")
  release.git(repo, "checkout", "main")
  if method == "squash":
    release.git(repo, "merge", "--squash", "feature")
    introducing = commit(repo, "squashed PR")
  elif method == "merge":
    release.git(repo, "merge", "--no-ff", "feature", "-m", "merged PR")
    introducing = release.git(repo, "rev-parse", "HEAD")
  else:
    release.git(repo, "merge", "--ff-only", "feature")
    introducing = release.git(repo, "rev-parse", "HEAD")
  assert automation["tag_releases"](
      repo, lambda sha: sha == introducing) == ["v2.0.1.0"]
  assert version(repo) == "2.0.1.0"


def test_missing_history_fails_instead_of_inventing_a_version(repo):
  release.git(repo, "tag", "-d", "v2.0.0.0")
  with pytest.raises(RuntimeError, match="baseline"):
    version(repo)


def test_shallow_checkout_fails_with_recovery_command(repo, tmp_path):
  clone = tmp_path / "shallow"
  subprocess.run(
      ["git", "clone", "--depth=1",
       repo.as_uri(), str(clone)],
      check=True,
      capture_output=True)
  with pytest.raises(RuntimeError, match="git fetch --unshallow --tags"):
    version(clone)


def test_linked_worktree_version(repo, tmp_path):
  worktree = tmp_path / "linked"
  release.git(repo, "worktree", "add", "-b", "linked", str(worktree))
  commit(worktree)
  assert version(worktree) == "2.0.0.1"


def test_artifact_version_needs_no_git(tmp_path):
  package = tmp_path / "postgkyl"
  package.mkdir()
  (package / release.VERSION_FILE).write_text("2.3.4.5\n")
  assert release.get_version(package) == "2.3.4.5"


def test_unversioned_source_archive_fails(tmp_path):
  with pytest.raises(RuntimeError, match="Git clone"):
    version(tmp_path)


def test_automation_catches_up_once_in_history_order(repo):
  first = commit(repo, "squashed PR")
  commit(repo, "direct push")
  second = commit(repo, "last commit of rebased PR")
  commit(repo, "direct push after PR")
  merged = lambda sha: sha in {first, second}
  tagged = automation["tag_releases"](repo, merged)
  assert tagged == ["v2.0.1.0", "v2.0.2.0"]
  assert release.git(repo, "rev-parse", "v2.0.1.0^{commit}") == first
  assert release.git(repo, "rev-parse", "v2.0.2.0^{commit}") == second
  assert version(repo) == "2.0.2.1"
  assert automation["tag_releases"](repo, merged) == []
  third = commit(repo, "another merged PR")
  assert automation["tag_releases"](repo,
                                    lambda sha: sha == third) == ["v2.0.3.0"]


@pytest.mark.parametrize(("bump", "expected"), [("minor", "2.1.0.0"),
                                                ("major", "3.0.0.0")])
def test_manual_release_resets_and_reruns(repo, bump, expected):
  commit(repo)
  tag_releases = automation["tag_releases"]
  assert tag_releases(repo, lambda _: True, bump,
                      "123") == ["v2.0.1.0", "v" + expected]
  assert version(repo) == expected
  commit(repo, "main moves before a workflow rerun")
  assert tag_releases(repo, lambda _: True, bump, "123") == []


@pytest.mark.parametrize("change", [{}, {
    "merged_at": None
}, {
    "merge_commit_sha": "other"
}, {
    "base": {
        "ref": "other",
        "repo": {
            "full_name": "owner/repo"
        }
    }
}])
def test_github_pr_classification(monkeypatch, change):
  pr = {
      "merged_at": "today",
      "merge_commit_sha": "abc",
      "base": {
          "ref": "main",
          "repo": {
              "full_name": "owner/repo"
          }
      }
  } | change
  function = automation["is_merged_pr"]
  monkeypatch.setitem(
      function.__globals__, "urlopen",
      lambda *args, **kwargs: io.BytesIO(json.dumps([pr]).encode()))
  assert function("owner/repo", "test-token", "abc") is (not change)


def test_github_pagination(monkeypatch):
  pr = {
      "merged_at": "today",
      "merge_commit_sha": "abc",
      "base": {
          "ref": "main",
          "repo": {
              "full_name": "owner/repo"
          }
      }
  }
  responses = iter([[pr | {"merged_at": None}] * 100, [pr]])
  urls = []

  def response(request, **kwargs):
    urls.append(request.full_url)
    return io.BytesIO(json.dumps(next(responses)).encode())

  function = automation["is_merged_pr"]
  monkeypatch.setitem(function.__globals__, "urlopen", response)
  assert function("owner/repo", "test-token", "abc")
  assert urls[-1].endswith("page=2")
