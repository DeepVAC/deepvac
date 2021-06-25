import setuptools
import sys
from deepvac import __version__
def readme():
    with open('README.md') as f:
        return f.read()

version=__version__
print("You are building deepvac with version: {}".format(version))
setuptools.setup(name='deepvac',
    version=version,
    description='PyTorch python project standard',
    long_description='PyTorch python project standard',
    keywords='PyTorch python project standard',
    url='https://github.com/DeepVAC/deepvac',
    download_url="",
    author='Gemfield',
    author_email='gemfield@civilnet.cn',
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_dir={
        'deepvac': 'deepvac',
    },

    install_requires=[
        'opencv_python',
        'numpy',
        'Pillow',
        'scipy',
        'tensorboard'
    ],
    classifiers  = [
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries"
     ],
    zip_safe=False)
