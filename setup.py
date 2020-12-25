
from setuptools import setup

setup(
   name='tureplicator',
   version='1.0.0',
   description='A module to replicate and generate another tumor in an image of the brain',
   author='Nicholas Law',
   author_email='nicholas_law_91@hotmail.com',
   packages=['tureplicator'],  #same as name
   install_requires=[
       "numpy==1.19.1",
       "SimpleITK==2.0.1",
       "scikit-image==0.17.2",
       "opencv-python==4.2.0.34",
       "matplotlib==3.3.0",
       "Pillow==7.0.0"
   ], #external packages as dependencies
)