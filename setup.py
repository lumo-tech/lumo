from setuptools import setup, find_packages
from lumo import __version__

# from rename_templete import copy_templete

# copy_templete()

"""
python3 setup.py sdist bdist_wheel; 
python3 setup.py sdist bdist_wheel; sudo pip install dist/$(python3 install.py);
python3 setup.py sdist bdist_wheel; pip install dist/$(python3 install.py) --user
sudo pip install dist/$(python3 install.py);
pip install dist/$(python3 install.py) --user
"""

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
        'fire', 'psutil', 'joblib'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='lumo',
    packages=['lumo'],
    entry_points={
        'console_scripts': [
            'lumo = lumo.cli.cli:main'
        ]
    },
)
