"""``evaluate`` -- evaluate an RPN math expression over the active datasets."""

from __future__ import annotations

import re

import click

import postgkyl as pg

from .._apply import active_datasets, set_active
from .._options import label_option, tag_option

_OPERATORS = ", ".join(pg.available_evaluate_operators())

_DATA_TOKEN = re.compile(r"^f(\d*)(?:\[[^\]]*\])?(?:\.\w+)?$")

_HELP = f"""Evaluate an RPN expression over the active datasets.

An expression using only bare ``f`` references is evaluated independently
for every active dataset, e.g. ``evaluate "f grad"`` takes the gradient of
each one. Explicit ``fN`` tokens combine datasets positionally, e.g.
``evaluate "f0 f1 +"``. The input datasets are deactivated and the result(s)
are appended to the working set; datasets already inactive are untouched.

Note: with ``chain=True``, ``--tag``/``--label`` must be given *before*
CHAIN (``evaluate --tag foo "f0 f1 +"``), not after -- see ``fit``'s docstring.

\b
Supported operators: {_OPERATORS}"""


@click.command("evaluate", help=_HELP)
@click.argument("chain")
@tag_option()
@label_option()
@click.pass_context
def command(ctx, chain, tag, label) -> None:
  pool = active_datasets(ctx)
  if not pool:
    raise click.UsageError("evaluate: no datasets to evaluate")
  # end
  try:
    references = [match for token in chain.split()
        if (match := _DATA_TOKEN.fullmatch(token)) is not None]
    map_over_pool = bool(references) and all(not match.group(1) for match in references)
    if map_over_pool:
      results = [pg.evaluate(chain, d, tag=tag, label=label) for d in pool]
    # end
    else:
      results = [pg.evaluate(chain, *pool, tag=tag, label=label)]
    # end
  # end
  except ValueError as err:
    raise click.UsageError(str(err))
  # end
  for d in pool:
    set_active(d, False)
  # end
  ctx.obj.datasets.extend(results)
# end
