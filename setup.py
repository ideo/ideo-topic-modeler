from distutils.core import setup

setup(
    name='ideo_topic_modeler',
    version='0.1dev',
    packages=['ideo_topic_modeler',],
    install_requires=[
        "pandas",
        "altair",
        "matplotlib",
        "sentence-transformers",
        "bertopic",
        "umap-learn",
      ],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
)