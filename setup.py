from setuptools import setup, find_packages


__author__ = 'Chris R. Vernon'
__email__ = 'chris.vernon@pnnl.gov'
__copyright__ = 'Copyright (c) 2017, Battelle Memorial Institute'
__license__ = 'BSD 2-Clause'


def readme():
    with open('README.md') as f:
        return f.read()


def get_requirements():
    with open('requirements.txt') as f:
        return f.read().split()


setup(
    name='clm_demeter',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/IMMM-SFA/clm_demeter.git',
    license='BSD2-Clause',
    author='Chris R. Vernon',
    author_email='chris.vernon@pnnl.gov',
    description='Preparatory code to integrate CLM and Demeter',
    long_description=readme(),
    install_requires=get_requirements(),
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, <4'
)
