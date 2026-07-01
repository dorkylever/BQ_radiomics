# coding: utf-8

from setuptools import setup, find_packages
from pathlib import Path

# Get __verison_dunder without importing BQ_radiomics
version_file = Path(__file__).resolve().parent / 'BQ_radiomics' / 'version.py'
exec(open(version_file).read())


setup(
    name='BQ-radiomics',
    download_url=f'https://github.com/dorkylever/BQ_radiomics.git',
    version="0.0.2",
    packages=find_packages(exclude=("dev")),
    package_data={},  # Puts it in the wheel dist. MANIFEST.in gets it in source dist
    include_package_data=True,
    install_requires=[
        'appdirs',
        'setuptools',
        'matplotlib>=2.2.0',
        'numpy>=1.23,<2',
        'pandas>=1.1.0',
        'scikit-learn',
        'scipy>=1.1.0',
        'scikit-image',
        'seaborn>=0.9.0',
        'PyYAML>=3.13',
        'catboost',
        'SimpleITK',
        'pyradiomics==3.1.0',
        'threadpoolctl',
        'imbalanced-learn',
        'raster-geometry',
        'filelock',
        'psutil',
        'plotly',
        'logzero==1.7.0',
        'numba',
        'addict',
        'toml',
        'pynrrd',
        'pytest',
        'tqdm',
        'gitpython',
        'pacmap==0.7.0',
        'shap',
        'joblib',
        'wheel',
        'numexpr',
        'bottleneck'
    ],
    extras_require={},
    url='https://github.com/dorkylever/BQ_radiomics',
    license='',
    author='Kyle Drover',
    author_email='kyle.drover@anu.edu.au',
    description='Perform Radiomics Feature Extraction and Machine Learning',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
     ],
    keywords=['radiomics'],
    entry_points ={
            'console_scripts': [
                'BQ_radiomics_runner=BQ_radiomics.scripts.lama_radiomics_runner:main',
                'BQ_radiomics_machine_learning=BQ_radiomics.scripts.lama_machine_learning:main',
                'BQ_radiomics_secondary_validation=BQ_radiomics.scripts.confusion_matrix_generator:main'
            ]
        },
)
