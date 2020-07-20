import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gradrack",
    version="0.0.1",
    author="SIGNAS",
    author_email="b.j.hayes@se19.qmul.ac.uk",
    description="A differentiable library of synthesiser components",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/signas-qmul/gradrack",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch==1.5.0"
    ]
)
