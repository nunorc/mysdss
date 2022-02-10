
from setuptools import setup, find_packages
from mysdss import __version__

with open('README.rst', 'r') as fh:
    long_description = fh.read()

setup(name = 'astromlp',
      version = __version__,
      url = 'https://github.com/nunorc/mysdss',
      author = 'Nuno Carvalho',
      author_email = 'narcarvalho@gmail.com',
      description = 'toolbox for sdss',
      long_description = long_description,
      long_description_content_type = 'text/x-rst',
      license = 'MIT',
      packages = find_packages(),
      install_requires = ['numpy','pandas', 'requests', 'tqdm', 'tensorflow', 'scikit-learn'])

