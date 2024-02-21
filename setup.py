from distutils.core import setup
from setuptools import find_packages

requires = [
    'six>=1.10.0',
]

if __name__ == "__main__":
    setup(
        name="qmc",
        version="0.0.1",
        packages=find_packages(),
        author='S. Molina & S. Quiroga & D. Useche',
        install_requires=requires,
        description='qml models',
        include_package_data=True,
    )