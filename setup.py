import setuptools
import sys
def readme():
    with open('README.md') as f:
        return f.read()

version="0.1.14"
setuptools.setup(name='deepvac',
    version=version,
    description='PyTorch python project standard',
    long_description='PyTorch python project standard',
    keywords='PyTorch python project standard',
    url='https://github.com/DeepVAC/deepvac',
    download_url="",
    author='Gemfield',
    author_email='gemfield@civilnet.cn',
    #packages=setuptools.find_packages(),
    packages=[
        'deepvac',
    ],
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
    include_package_data=True,
    zip_safe=False)
