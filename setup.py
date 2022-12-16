#!/usr/bin/env python
from setuptools import find_packages, setup

if __name__ == '__main__':
    setup(name='visibility_detection',
          version='0.1.0',
          description='Visibility Detection',
          packages=find_packages(),
          install_requires=[
              'importlib_resources',
              'Pillow',
              'scikit-learn',
              'scipy',
              'tensorboardX',
              'pandas',
              'torch',
              'torchvision',
          ])
