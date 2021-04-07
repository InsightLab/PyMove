"""Package setup."""

from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

with open('requirements.txt') as f:
    DEPENDENCIES = f.readlines()

setup(
    name='pymove',
    version='2.7.0',
    author='Insight Data Science Lab',
    author_email='insightlab@dc.ufc.br',
    license='MIT',
    python_requires='>=3.6',
    description='A lib python to processing and visualization '
                'of trajectories and other spatial-temporal data',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/InsightLab/PyMove',
    packages=find_packages(exclude=['*.tests']),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Typing :: Typed'
    ],
    install_requires=DEPENDENCIES,
    include_package_data=True
)
