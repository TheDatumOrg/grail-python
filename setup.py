import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="grailts",
    version="0.0.1",
    author="Karhan Kaan Kayan",
    author_email="kayankarhankaan@gmail.com",
    description="Python implementation of GRAIL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/karhankaan/GRAIL",
    project_urls={
        "Bug Tracker": "https://github.com/karhankaan/GRAIL/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    install_requires=['numpy', 'scipy', 'tslearn'],
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)