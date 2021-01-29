git pull
python3 setup.py sdist bdist_wheel
pip3 install $(python3 install.py)