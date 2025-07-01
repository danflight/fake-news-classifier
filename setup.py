from setuptools import setup, find_packages

# with open("README.md") as f:
#     LONG_DESCRIPTION = f.read()

# def get_install_requires():
#     with open("requirements.txt", "r") as f:
#         return [line.strip() for line in f.readlines() if not line.startswith("-")]

setup(
    name="fake-news-classifier", 
    version="0.1",
    license="MIT",
    author="Daniel Flight",
    author_email="danielflight07@gmail.com",
    description="",
    # long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    # install_requires=get_install_requires(),
    packages=find_packages(exclude=("tests","bechmarks","examples")),
    python_requires=">=3.10"
)

