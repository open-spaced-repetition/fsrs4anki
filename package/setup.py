import setuptools

with open("requirements.txt", "r") as f:
    requires = [a.strip('\n') for a in f.readlines()]

setuptools.setup(
    name="fsrs4anki_optimizer",
    version="3.19.0",
    packages=setuptools.find_packages(),
    install_requires=requires
)