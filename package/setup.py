import setuptools
import sys

version = sys.argv.pop(2)

with open("requirements.txt", "r") as f:
    requires = [a.strip('\n') for a in f.readlines()]

setuptools.setup(
    name="fsrs4anki_optimizer",
    version=version,
    packages=setuptools.find_packages(),
    install_requires=requires
)