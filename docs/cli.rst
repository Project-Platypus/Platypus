==================
Command Line Tools
==================

Starting wth version 1.4.0, Platypus offers several command-line tools.  To
see all available options, run:

.. code:: bash

   python -m platypus --help

Solve
-----

Solve executes the given algorithm on a problem, printing the final approximation set as JSON:

.. code:: bash

   python -m platypus solve --algorithm NSGAII --problem DTLZ2 --nfe 10000

Additional arguments can be provided, however, note this only works with built-in Python types like
``str``, ``int`` and ``float``:

.. code:: bash

   python -m platypus solve --algorithm NSGAII --problem DTLZ2 --nfe 10000 populationSize=250

We can also save the output to a file for further analysis:

.. code:: bash

   python -m platypus solve --algorithm NSGAII --problem DTLZ2 --nfe 10000 --output NSGAII_DTLZ2.set

Performance Indicators
----------------------

Now that we have the output from a run, we can compute any performance indicator.  Here, we calculate
the hypervolume using a reference set for normalization:

.. code:: bash

   python -m platypus hypervolume --reference ./examples/DTLZ2.2D.pf NSGAII_DTLZ2.set

::

   0.23519328623761482

Plotting
--------

For 2 and 3 objectives, we can also generate a plot, either interactive or saving as an image:

.. code:: bash

   python -m platypus plot NSGAII_DTLZ2.set