from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='aklearn', 
    version='1.0', 
    description = 'A ML library built from scratch',
    author = 'Akshay Prasadan',
    author_email = 'akprasadan@gmail.com',
    license='LICENSE.txt',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url='https://github.com/akprasadan/aklearn.git',
    keywords='machine learning, data science, statistics, numpy, python',
    classifiers = [
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8"
    ])
