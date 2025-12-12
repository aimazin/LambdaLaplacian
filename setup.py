from setuptools import setup, find_packages

setup(
    name='LambdaLaplacian',
    version='0.1.0',
    description='Lagrangian-style imputation with Laplacian regularization',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy>=1.21',
        'scipy>=1.7',
        'matplotlib>=3.0'
    ],
    python_requires='>=3.8',
    license='Apache-2.0',
    author='aimazin'
)
