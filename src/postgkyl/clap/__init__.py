"""High-level, scriptable pgkyl CLAP.

Re-exports the generated :class:`PgkylSession` so it can be imported directly
from the package:

    from postgkyl.clap import PgkylSession
"""

from postgkyl.clap.clap import PgkylSession

__all__ = ["PgkylSession"]
