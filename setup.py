import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()
    for i, r in enumerate(requirements):
        requirements[i] = r.replace("\n", "")

if __name__ == "__main__":
    setuptools.setup(
        name="pylgr",
        version="0.3.2",
        description="Legendre-Gauss-Radau pseudospectral method for optimal control",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Tenavi Nakamura-Zimmerer",
        author_email="tenavi.nakamura-zimmerer@nasa.gov",
        packages=["pylgr"],
        install_requires=requirements
    )