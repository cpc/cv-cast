# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

packages = \
['torchjpeg',
 'torchjpeg.codec',
 'torchjpeg.data',
 'torchjpeg.data.transforms',
 'torchjpeg.dct',
 'torchjpeg.dct.stats',
 'torchjpeg.metrics',
 'torchjpeg.quantization']

package_data = \
{'': ['*']}

install_requires = \
['Pillow>=7.2.0', 'torch>=2.0.0,<2.0.1', 'torchvision>=0.15.1,<0.15.2']

setup_kwargs = {
    'name': 'torchjpeg',
    'version': '0.0.0',
    'description': 'Utilities for JPEG data access and manipulation in pytorch',
    'long_description': '# TorchJPEG\n\n[![pipeline status](https://gitlab.com/torchjpeg/torchjpeg/badges/master/pipeline.svg)](https://gitlab.com/torchjpeg/torchjpeg/-/commits/master) \n[![coverage report](https://gitlab.com/torchjpeg/torchjpeg/badges/master/coverage.svg)](https://gitlab.com/torchjpeg/torchjpeg/-/commits/master) \n[![PyPI](https://img.shields.io/pypi/v/torchjpeg)](https://pypi.org/project/torchjpeg/)\n[![License](https://img.shields.io/badge/license-MIT-blue)](https://gitlab.com/Queuecumber/torchjpeg/-/blob/master/LICENSE)\n\n_Custom fork to experiment with using custom YCbCr transform and per-channel quantization tables._\n\nThis package contains a C++ extension for pytorch that interfaces with libjpeg to allow for manipulation of low-level JPEG data.\nBy using libjpeg, quantization results are guaranteed to be consistent with other applications, like image viewers or MATLAB,\nwhich use libjpeg to compress and decompress images. This is useful because JPEG images can be effected by round-off\nerrors or slight differences in the decompression procedure. Besides this, this library can be used to read and write\nDCT coefficients, functionality which is not available from other python interfaces.\n\nBesides this, the library includes many utilities related to JPEG compression, many of which are written using native pytorch code meaning\nthey can be differentiated or GPU accelerated. The library currently includes packages related to the DCT, quantization, metrics, and dataset\ntransformations.\n\n## LIBJPEG\n\nCurrently builds against: `libjpeg-9d`. libjpeg is statically linked during the build process. See [http://www.ijg.org/files/](http://www.ijg.org/files/) for libjpeg source. \nThe full libjpeg source is included with the torchjpeg source code and will be built during the install process (for a source or sdist install).\n\n## Install\n\nPackages are hosted on [pypi](https://pypi.org/project/torchjpeg/). Install using pip, note that only Linux builds are supported at the moment. \n\n```\npip install torchjpeg\n```\n\nIf there is demand for builds on other platforms it may happen in the future. Also note that the wheel is intended to be compatible with manylinux2014\nwhich means it should work on modern Linux systems, however, because of they way pytorch works, we can\'t actually build it using all of the manylinux2014\ntools. So compliance is not guaranteed and YMMV.\n\n```{warning}\ntorchjpeg is currently in pre-beta development and consists mostly of converted research code. The public facing API, including any and all names of\nparameters and functions, is subject to change at any time. We follow semver for versioning and will adhere to that before making and breaking\nchanges.\n```\n\n## Citation\n\nIf you use our code in a publication, we ask that you cite the following paper ([bibtex](http://softmax.me/bibtex/ehrlich2020quantization.txt)):\n\n> Max Ehrlich, Larry Davis, Ser-Nam Lim, and Abhinav Shrivastava. "Quantization Guided JPEG Artifact Correction." In Proceedings of the European Conference on Computer Vision, 2020',
    'author': 'Max Ehrlich',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://torchjpeg.readthedocs.io',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<4.0',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
