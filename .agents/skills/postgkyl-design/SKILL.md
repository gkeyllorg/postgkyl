---
name: postgkyl-design
description: Apply Postgkyl coding doctrine when designing, implementing, or reviewing Python changes.
---

# Coding Doctrine

**0. Locality of reasoning.** Every principle below is a projection of
one axiom: a reader must be able to understand a fragment without the
whole program. Whatever keeps a local conclusion sound — a frozen
record, an honest signature, a stated law — is doctrine. Whatever
forces a global search — ambient state, a leaky layer, a second copy
of a fact — is the enemy.

*Data — what it does, and what it may say*

**I. Data is inert. Functions transform.** No objects that know
things and do things. Data is a frozen record. Behavior is a function
that takes data in and returns data out. If you're reaching for
inheritance, you've taken a wrong turn.

**II. Make illegal states unrepresentable.** The shape of a datum is
its strongest invariant. Constructors refuse invalid states; a checked
fact becomes a type; downstream never re-proves what upstream
established. Parse, don't validate.

*Functions — one idea, honestly declared*

**III. A function is one idea.** It takes exactly what it needs and
returns exactly what it computes. If the signature has two concepts in
it, you have two functions.

**IV. The signature tells the whole truth.** Inward: if something
needs a value, it receives it as a parameter — no spooky action at a
distance, no stringly-typed interfaces, no implicit state. Outward:
same inputs, same outputs; effects and failure appear in the type, not
in the fine print. Pure core, effects at the edges.

*Knowledge — one home per fact*

**V. Every fact has one home.** One authoritative representation of
each decision and each piece of knowledge; everything else inherits or
is derived mechanically — never maintained by hand in parallel.
Configuration is decided once, at the highest level, and threaded
down; no module ever decides its own context. If the design and the
implementation can disagree, you have two sources of truth and zero.

*Layers — what above, how below*

**VI. Separate what from how.** Logic and machinery are different
concerns with a hard boundary. The layer that says *what* to compute
should be readable by someone who has never seen the machinery
underneath. The layer that says *how* lives below, stays below, and
nothing leaks up from it.

**VII. Notation is execution; lowering is transliteration.** Looking
up: the spec layer reads like the math or logic it implements — when
notation *is* the executable object, not a comment beside it, bugs
have nowhere to hide. Looking down: the layer that executes the spec
reproduces it exactly — nothing added, nothing dropped, nothing
reinterpreted; no opinions, no defaults, no helpful conversions. If
the lowering changes anything, the spec is a lie.

*Abstraction — earned, and binding*

**VIII. Earn your abstractions.** No abstraction before the second
use. Three similar lines is better than a premature helper. The right
amount of complexity is the minimum the current task demands — not the
current task plus three hypothetical future ones.

**IX. An abstraction is a contract.** It is defined by what it
guarantees, not what it hides. If you can't state what is always true
of it — properties a client may rely on without reading the
implementation — it isn't an abstraction, it's indirection. Two
implementations that honor the contract must be interchangeable; and
its outputs stay in its vocabulary, so uses compose.

*Verification — formal first*

**X. Trust the most formal thing first.** Types over tests, tests
over docs, docs over comments. Invest in whichever layer catches the
bug earliest with the least ongoing maintenance cost.
