from setuptools import setup

setup(
    name="tslearn",
    description="A machine learning toolkit dedicated to time-series data",
    packages=['tslearn'],
    package_data={"tslearn": [".cached_datasets/Trace.npz"]},
    data_files=[("", ["LICENSE"])],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'Cython', 'hyperas'],
    version='0.1.28.3',
    url="http://tslearn.readthedocs.io/",
    author="Romain Tavenard",
    author_email="romain.tavenard@univ-rennes2.fr"
)
