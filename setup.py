#!/usr/bin/env python

from setuptools import setup

setup(name='Platypus-Opt',
      version='1.0.4', # Update __init__.py if the version changes!
      description='Multiobjective optimization in Python',
      author='David Hadka',
      author_email='dhadka@users.noreply.github.com',
      license="GNU GPL version 3",
      url='https://github.com/Project-Platypus/Platypus',
      packages=['platypus'],
      install_requires=['six'],
      tests_require=['pytest', 'mock'],
      python_requires='>=3.6'
     )
