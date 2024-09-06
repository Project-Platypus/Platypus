======================
Decsion Variable Types
======================

Real
----

Real valued variables are expressed as floating-point numbers between
some minimum and maximum bounds.  For example, in the following, we
configure the bounds as ``[-10, 10]``:

.. literalinclude:: ../examples/custom_problem_function.py
   :language: python

Binary
------

Binary variables store a "binary string".  Traditionally, this is
represented as ``0`` and ``1``, but in Platypus, it is stored as an
array of ``False`` and ``True`` values.  The knapsack problem demonstrates
the binary representation, where an item is selected when the binary
value is ``True``.  The goal, then, is to maximize the profit of the
selected items while being constrained by the total capacity.

.. literalinclude:: ../examples/knapsack.py
   :language: python

Integer
-------

The integer variable is a mix between ``Real`` and ``Binary``.  The variable
is constructed with minimum and maximum bounds, but is internally represented
as a Gray-coded binary variable.  The Gray-coding ensures each adjacent
integer value can be produced by a single bit mutation.

Permutation
-----------

The permutation variable creates a permutation of a list of items.  That is,
each item must appear in the list, but the ordering can change.  Here we
construct a permutaton of the numbers ``0`` through ``10`` using the ``range``
functon.  However, note that the items is not limited to a list of integers,
any collection of objects can be provided.

.. literalinclude:: ../examples/permutation.py
   :language: python

Subset
------

Lastly we have the subset.  Similar to the mathematical ``choose(n, k)`` operation,
a subset variable represents a fixed-size selection of size ``k`` from a
collection of ``n`` items.
