"""Tests for postgkyl.render.labels -- latex_to_unicode / latex_to_html."""

from __future__ import annotations

from postgkyl.render.labels import latex_to_html, latex_to_unicode


class TestLatexToUnicode:

  def test_empty_string_passthrough(self):
    assert latex_to_unicode("") == ""

  def test_plain_text_unchanged(self):
    assert latex_to_unicode("hello") == "hello"

  def test_strips_dollar_delimiters(self):
    assert latex_to_unicode(r"$\mu$") == "μ"

  def test_greek_letter_without_dollars(self):
    assert latex_to_unicode(r"\rho") == "ρ"

  def test_multiple_greek_letters(self):
    assert latex_to_unicode(r"\alpha \beta \gamma") == "α β γ"

  def test_uppercase_greek_letters(self):
    assert latex_to_unicode(
        r"\Omega \Delta \Theta \Sigma \Lambda") == "Ω Δ Θ Σ Λ"

  def test_parallel_and_perp_with_subscripts_unconverted(self):
    assert latex_to_unicode(r"$\mu_{\parallel}$") == "μ_{∥}"
    assert latex_to_unicode(r"E_{\perp}") == "E_{⊥}"

  def test_strips_surrounding_whitespace(self):
    assert latex_to_unicode("  \\pi  ") == "π"


class TestLatexToHtml:

  def test_empty_string_passthrough(self):
    assert latex_to_html("") == ""

  def test_plain_text_unchanged(self):
    result = latex_to_html("field")
    assert result == "field"

  def test_brace_subscript_becomes_html_sub(self):
    result = latex_to_html(r"$B_{x}$")
    assert result == "B<sub>x</sub>"

  def test_bare_subscript_becomes_html_sub(self):
    result = latex_to_html("n_0")
    assert result == "n<sub>0</sub>"

  def test_greek_letter_converted(self):
    result = latex_to_html(r"$\omega$")
    assert "ω" in result

  def test_greek_and_subscript_combined(self):
    assert latex_to_html(r"$\mu_{\parallel}$") == "μ<sub>∥</sub>"
    assert latex_to_html(r"E_{\perp}") == "E<sub>⊥</sub>"
