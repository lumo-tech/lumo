import sys

from setuptools import setup, find_packages

"""
python3 setup.py sdist bdist_wheel; 
python3 setup.py sdist bdist_wheel; pip3 install dist/$(python3 install.py);
python3 setup.py sdist bdist_wheel; pip install dist/$(python3 install.py) --user
python3 setup.py sdist bdist_wheel; pip install dist/$(python3 install.py) 
python3 setup.py sdist bdist_wheel; pip3 install dist/$(python3 install.py) 
sudo pip install dist/$(python3 install.py);
pip install dist/$(python3 install.py) --user
"""

print(find_packages('src'))

setup(
    name='lumo',
    version="0.14.0",
    description='library to manage your pytorch experiments.',
    long_description_content_type='text/markdown',
    url='https://github.com/sailist/lumo',
    author='sailist',
    author_email='sailist@outlook.com',
    license='Apache License 2.0',
    include_package_data=True,
    install_requires=[
        'fire', 'psutil', 'joblib', 'accelerator'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    package_dir={"": "src"},
    keywords='lumo',
    packages=find_packages('src'),
    entry_points={
        'console_scripts': [
            'lumo = lumo.cli.cli:main'
        ]
    },
)
