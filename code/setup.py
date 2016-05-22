try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(
  name = 'TensorCompletion2',
  version = '1.0',
  packages = find_packages(exclude = ['tests']),
  scripts = [],

  install_requires = ['numpy>=1.10.1', 'multiprocessing', ],
  
  author = 'Jingshu Liu',
  author_email = 'jl7722@nyu.edu',
  description = 'Scripts for tensor completion project'

)

# Note:
# Also require to install tensorflow. Follow instructions in 
# https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#download-and-setup