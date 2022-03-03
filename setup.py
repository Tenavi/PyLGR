import setuptools

# To install in developer mode run "pip install -e ."
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pylgr",
    version="0.1.0",
    description="Legendre-Gauss-Radau pseudospectral method for optimal control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tenavi Nakamura-Zimmerer",
    author_email="tenakamu@ucsc.edu",
    packages=["pylgr"]
)
