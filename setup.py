from setuptools import setup, find_packages
from os import path
working_directory = path.abspath(path.dirname(__file__))

with open(path.join(working_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='abas_ml_lib', # name of packe which will be package dir below project
    version='0.0.1',
    url='https://github.com/Mti-Abas/Abas_ML_Lib',
    author='Abas Ahmed',
    author_email='basaltmymy0@gmai.com',
    description='Simple test package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[],
)
