from setuptools import setup, find_packages

setup(
    name="gega",
    version="0.1.0",
    description=("A generalised genetic algorithm implementation"),
    packages=find_packages(),
    url="https://github.com/gordon-frost-hwu/gega.git",
    author="Gordon Frost",
    author_email="",
    install_requires=[
        "numpy",         # math library
        "pandas",       # plotting library
        "unittest",     # unit testing framework
    ]
)
