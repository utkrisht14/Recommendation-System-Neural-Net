from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="MLOps_Project_2",
    version="0.1",
    author="Utkrisht",
    packages=find_packages(),
    install_requires=requirements,
)