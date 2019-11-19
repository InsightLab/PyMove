from setuptools import setup

dependencies = ['numpy', 'pandas', 'folium', 'matplotlib', 'sharpely', '']

setup(name='pymove',
      version='0.1',
      description='A lib python to ',
      url='http://github.com/storborg/funniest',
      author='Flying Circus',
      author_email='flyingcircus@example.com',
      license='MIT',
      packages=['funniest'],
      install_requires=dependencies,
      zip_safe=False)