from setuptools import setup, find_packages

requirements = [
    'jupyter',
    'scikit-learn',
    'matplotlib',
    'pandas'
]

dev_requirements = [
    'pip-tools',
]

setup(
    name='machine_learning_ass1',
    version='0.0.0',
    description='Assignment #1 for COSC401-19S2: Machine Learning.',
    author='Alex Tompkins',
    author_email='ato47@uclive.ac.nz',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements
    }
)
