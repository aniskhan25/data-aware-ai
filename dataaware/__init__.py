"""Shared library for the Data-Aware AI on LUMI tutorial.

The command line entry points live in ``scripts/``. Everything that needs to be
tested or reused across experiments lives here:

``config``    strict YAML experiment configuration
``manifest``  the sample manifest shared by every dataset representation
``schema``    the machine-readable run-summary schema
``generate``  deterministic synthetic dataset generation
``metrics``   small statistics helpers used by the benchmarks
``env``       run context (hostname, Slurm, Git, peak memory)
"""

__version__ = "0.1.0"
