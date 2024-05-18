"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: NEUROWORK Research Labs
"""

from setuptools import setup, find_packages

setup(
    name='NRL-brain-tumour-segmentation', # name of the package
    version='1.0.0', # version of the package
    description='Performance analysis of novel unet architectures created using PyTorch framework on BraTS dataset',
    author='Yagnik Poshiya', # author of the package
    author_email='yagnikposhiya.research@gmail.com', # package author mail
    url='', # url of package repository
    packages=find_packages(), # automatically find packages in 'src' directory
    install_requires=[
        'torch',
        'wandb',
        'nilearn',
        'lightning'
        'torchvision',
        'https://github.com/miykael/gif_your_nifti'
    ], # list of dependencies required by the package
    classifiers=[
        'Programming Language :: Python :: 3.12.3'
    ] # list of classifiers describing package
)
