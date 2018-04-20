#!/usr/bin/env python

from setuptools import setup
from setuptools.command.test import test as TestCommand

class NoseTestCommand(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import nose
        nose.run_exit(argv=['nosetests'])

setup(name='Platypus-Opt',
      version='1.0.2', # Update __init__.py if the version changes!
      description='Multiobjective optimization in Python',
      author='David Hadka',
      author_email='dhadka@users.noreply.github.com',
      license="GNU GPL version 3",
      url='https://github.com/Project-Platypus/Platypus',
      packages=['platypus'],
      install_requires=['six'],
      tests_require=['nose', 'mock'],
      cmdclass={'test': NoseTestCommand},
     )