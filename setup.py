from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Streaming video data via networks'
LONG_DESCRIPTION = 'A package that allows to build simple 3d object obj files.'

# Setting up
setup(
    name="pythreemaker",
    version=VERSION,
    author="rootjatin (Jatin Sharma)",
    author_email="<jatin100198@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy'],
    keywords=['python', 'three dimensional', 'render', 'pythreemaker', '3d', 'blender'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Designers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
