
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='ProgettoMachineLearning',
    url='https://github.com/Andrew-Wyn/Neural_Network_didactic',
    author='Luca Moroni, Maria Cassese, Simone Manti',
    author_email='...',
    # Needed to actually package something
    packages=['mlprj'],
    # Needed for dependencies
    install_requires=required,
    include_package_data=True,
    package_data={'': ['data/*']},
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='A simple neural network simulator, implemented for an university exam',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
