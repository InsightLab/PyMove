try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# Get the long description from the README file
try:
    long_description = open("README.md", "r").read()
except Exception as e:
    raise e

import versioneer

cmdclass = versioneer.get_cmdclass()

DEPENDENCIES = [
    "tqdm",
    "numpy",
    "pandas",
    "scipy",
    "geojson",
    "matplotlib",
    "shapely",
    "branca",
    "folium>=0.10.1",
    "mplleaflet",
    "matplotlib",
    "psutil",
    "resource",
    "dask[dataframe]",
    "sklearn",
    "joblib",
]

setup(
    name="pymove",
    version=versioneer.get_version(),
    author="Insight Data Science Lab",
    author_email="insightlab@dc.ufc.br",
    license="MIT",
    python_requires=">=3.6",
    description="A lib python to processing and visualization of trajectories and other spatial-temporal data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/InsightLab/PyMove",
    packages=[
        "pymove",
        "pymove.core",
        "pymove.models",
        "pymove.models.pattern_mining",
        "pymove.osm_module",
        "pymove.preprocessing",
        "pymove.uncertainty",
        "pymove.utils",
        "pymove.visualization",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=DEPENDENCIES,
)
