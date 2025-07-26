from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    """
    This function reads the requirements file and returns a list of packages.
    """
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name='Hybrid_ML_model',
    version='0.1.0',
    author='Kaushik',
    author_email='kaushikgowda547@gmail.com',
    description='A machine learning project',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
