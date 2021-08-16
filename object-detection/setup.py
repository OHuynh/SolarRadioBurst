"""Setup script for object_detection."""

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['tensorflow==2.3.1', 'absl-py==0.12.0', 'astunparse==1.6.3', 'attrs==21.2.0', 'cached-property==1.5.2', 'certifi==2020.12.5', 'cffi==1.14.5', 'chardet==4.0.0', 'cycler==0.10.0', 'dm-tree==0.1.6', 'flatbuffers==1.12', 'future==0.18.2', 'dill==0.3.3', 'Pillow>=1.0', 'gast==0.4.0', 'gin-config==0.4.0', 'Matplotlib==3.4.2', 'Cython>=0.29.23', 'lvis', 'pycocotools', 'tf-models-official', 'numpy==1.19.5', 'joblib==1.0.1', 'tensorboard==2.5.0']

setup(
    name='object_detection',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('object_detection')],
    description='Tensorflow Object Detection Library',
)