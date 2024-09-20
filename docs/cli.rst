==================
Command Line Tools
==================

Starting wth version 1.4.0, Platypus offers several command-line tools.  To
see all available options, run:

.. code:: bash

   python -m platypus --help

Solve
-----

Solve executes the given algorithm on a problem, saving the final approximation set as JSON:

.. code:: bash

   python -m platypus solve --algorithm NSGAII --problem DTLZ2 --nfe 10000 --output NSGAII_DTLZ2.set

Additional arguments can be provided, such as setting the ``population_size`` as demonstrated below.
Note that only arguments with primitive types are supported.

.. code:: bash

   python -m platypus solve --algorithm NSGAII --problem DTLZ2 --nfe 10000 population_size=250

::

   INFO:Platypus:Setting population_size=250 on NSGAII
   INFO:Platypus:NSGAII starting
   INFO:Platypus:NSGAII finished; Total NFE: 10000, Elapsed Time: 0:00:06.216727

Performance Indicators
----------------------

Now that we have the output from a run, we can compute any performance indicator.  Here, we calculate
the hypervolume using a reference set for normalization:

.. code:: bash

   python -m platypus hypervolume --reference_set ./examples/DTLZ2.2D.pf NSGAII_DTLZ2.set

which outputs:

::

   0.23519328623761482

Filtering
---------

The results can be filtered to remove any dominated, infeasible, or duplicate solutions:

.. code:: bash

   python -m platypus filter --output NSGAII_DTLZ2_unique.set --feasible --unique NSGAII_DTLZ2.set
   python -m platypus filter --output NSGAII_DTLZ2_epsilons.set --epsilons 0.01,0.01 NSGAII_DTLZ2.set

Normalization
-------------

Most performance indicators are normalized, meaning the objectives are scaled by some
minimum and maximum bounds.  This is useful to make results comparable across different
studies, assuming the same reference set or bounds are supplied.  Here, we can generate
the normalized solutions:

.. code:: bash

   python -m platypus normalize --output NSGAII_DTLZ2_normalized.set --reference_set ./examples/DTLZ2.2D.pf NSGAII_DTLZ2.set
   python -m platypus normalize --output NSGAII_DTLZ2_normalized.set --minimum 0.0,0.0 --maximum 1.0,1.0 NSGAII_DTLZ2.set

Plotting
--------

For 2 and 3 objectives, we can also generate a plot, either interactive or saving as an image:

.. code:: bash

   python -m platypus plot NSGAII_DTLZ2.set
   python -m platypus plot --output NSGAII_DTLZ2.png NSGAII_DTLZ2.set

Combining or Chaining Commands
------------------------------

On systems that support piping the output from one command into another (i.e., ``cmd1 | cmd2``),
we can also use this to combine these CLI tools.  Simply exclude the input and output filenames,
as demonstrated below:

.. code:: bash

   python -m platypus solve --algorithm NSGAII --problem DTLZ2 --nfe 10000 | \
       python -m platypus filter --epsilons 0.01,0.01 | \
       python -m platypus plot
