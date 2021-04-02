def copy_templete():
    import os
    import shutil

    src = 'lumo-templete'
    dst = 'lumo/cli/templete'
    if os.path.exists(dst):
        shutil.rmtree(dst)

    shutil.copytree(src, dst)

    for root, dirs, fs in os.walk(dst):
        for f in fs:
            if f.endswith('.py'):
                f = os.path.join(root, f)
                prefn, ext = os.path.splitext(f)

                os.rename(f, '{}.py-tpl'.format(prefn))
