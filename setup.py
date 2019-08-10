try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

# Get the long description from the README file
try:
    long_description = open('README.md', "r").read()
except Exception as e:
    raise e
  
DEPENDENCIES = ['tqdm','numpy', 'pandas', 'scipy', 'geojson', 'matplotlib', 'shapely', 'folium', 'mplleaflet', 'psutil', 'ipywidgets']  

setup(
     name="pymove",  
     version="0.0.7",
     author="Insight Data Science Lab",
     author_email="nickssonarrais@insight.ufc.br",
     license='MIT',
     python_requires='>=3.6',
     description="PyMove to processing and visualization of trajectories and other spatial-temporal data.",
     long_description=long_description,
     url="https://github.com/InsightLab/PyMove",
     packages=find_packages(),
     classifiers=[
            'Development Status :: 4 - Beta',
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
     ],
     install_requires=DEPENDENCIES,
 )