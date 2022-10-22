import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setuptools.setup(
        name="pylgr",
        version="0.3.1",
        description="Legendre-Gauss-Radau pseudospectral method for optimal control",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Tenavi Nakamura-Zimmerer",
        author_email="tenakamu@ucsc.edu",
        packages=["pylgr"],
        install_requires=[
            "scipy>=1.5.2",
            "numpy>=1.19.1",
            "pytest>=6.1.1",
            "matplotlib>=3.1.2"
        ]
    )
