from setuptools import find_packages, setup

setup(
    name='bops',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy==1.22.3',
        'scipy==1.8.0',
        'numpy_ml==0.1.2'
    ],
    version='0.1',
    description='Boltzmann machines for Protein Signalling',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author='Nicolas Deutschmann',
    author_email="nicolas.deutschmann@gmail.com",
    include_package_data=True
)
