from lumo.exp import Experiment
import os

import zipfile


def test_extract(test_root, output=None, verbose=True):
    exp = Experiment.from_disk(test_root)
    if output is None:
        output = os.path.join(os.getcwd(), f'{exp.test_name}.zip')

    z = zipfile.ZipFile(output, 'a', zipfile.ZIP_DEFLATED)

    if verbose:
        print('deflate info')
    for root, dirs, fs in os.walk(exp.test_root):
        for f in fs:
            a = os.path.join(root, f)
            b = os.path.join('/', exp.exp_name, exp.test_name, root.replace(exp.test_root, 'experiment').lstrip('/'), f)
            z.write(a, b
                    )
            if verbose:
                print(f'{a} => {b}')

    if verbose:
        print('deflate blob')
    for root, dirs, fs in os.walk(exp.blob_root):
        for f in fs:
            a = os.path.join(root, f)
            b = os.path.join('/', exp.exp_name, exp.test_name, root.replace(exp.blob_root, 'blob/'), f)
            z.write(a, b
                    )
            if verbose:
                print(f'{a} => {b}')
