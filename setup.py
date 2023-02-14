import re
from datetime import datetime
from setuptools import setup, find_packages


def extract_version():
    return re.search(
        r'__version__ = "([\d.d\-]+)"',
        open('src/lumo/__init__.py', 'r', encoding='utf-8').read()).group(1)


if __name__ == '__main__':
    setup(
        name='lumo',
        version=extract_version(),
        description='library to manage your pytorch experiments.',
        long_description_content_type='text/markdown',
        url='https://github.com/pytorch-lumo/lumo',
        author='sailist',
        author_email='sailist@outlook.com',
        license_files=('LICENSE',),
        include_package_data=True,
        classifiers=[
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
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
