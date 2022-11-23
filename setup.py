try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='tools',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description="Environment for Team Andrew.",
      long_description="Environment for team Andrew.",
      packages=['tools'])
