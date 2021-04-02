from setuptools import setup, find_packages
from lumo.version import __version__
from rename_templete import copy_templete

copy_templete()

setup(
    name='lumo',
    version=__version__,
    description='torch kit for programing your dl experiments code elegant.',
    url='https://github.com/sailist/lumo',
    author='sailist',
    author_email='sailist@outlook.com',
    license='Apache License 2.0',
    include_package_data=True,
    install_requires=[
        'torch',
        'matplotlib', 'numpy==1.18.0', 'pandas',
        'fire','psutil',
        'gitpython',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='lumo',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'lumo = lumo.cli.cli:main'
        ]
    },
)
