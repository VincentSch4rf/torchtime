from setuptools import setup

setup(name='torchtime',
      version='0.01',
      description='A PyTorch Library for Time Series Classification',
      url='http://github.com/vincentsch4rf/torchtime',
      author='Vincent Scharf',
      author_email='vincent.scharf@smail.inf.h-brs.de',
      license='Apache2',
      packages=['torchtime',
                'torchtime.io',
                'torchtime.datasets',
                'torchtime.transforms',
                'torchtime.models'],
      zip_safe=False)
