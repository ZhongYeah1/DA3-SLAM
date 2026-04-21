from setuptools import setup, find_packages

setup(
    name='da3_slam',
    version='1.0.0',
    description='A feedforward SLAM system using Depth-Anything-3, optimized on the SL(4) manifold.',
    packages=find_packages(include=['evals', 'evals.*', 'da3_slam', 'da3_slam.*']),
)
