"""The generated session stub has the repository's canonical YAPF format."""

import sys

import pytest

from postgkyl.cli import session
from scripts import generate_session_stub


def test_generator_formats_once_and_updates_stale_stub(tmp_path, monkeypatch,
                                                       capsys):
  stub = tmp_path / "session.pyi"
  generated = "class Example:\n    def method(self) -> int:\n        ...\n"
  stub.write_text("class Example:\n  def old(self):\n    ...\n")
  monkeypatch.setattr(generate_session_stub, "STUB", stub)
  monkeypatch.setattr(session, "render_stub", lambda: generated)
  monkeypatch.setattr(sys, "argv", ["generate_session_stub.py"])

  generate_session_stub.main()
  assert stub.read_text() == generate_session_stub.format_stub(generated)
  assert "wrote" in capsys.readouterr().out

  generate_session_stub.main()
  assert "up to date" in capsys.readouterr().out

  updated = generated.replace("-> int", "-> str")
  monkeypatch.setattr(session, "render_stub", lambda: updated)
  generate_session_stub.main()
  assert stub.read_text() == generate_session_stub.format_stub(updated)
  assert "wrote" in capsys.readouterr().out


def test_generator_rejects_a_different_yapf_version(monkeypatch):
  monkeypatch.setattr(generate_session_stub, "version", lambda _: "0.31.0")
  with pytest.raises(RuntimeError, match="requires YAPF 0.43.0"):
    generate_session_stub.format_stub("x = 1\n")
