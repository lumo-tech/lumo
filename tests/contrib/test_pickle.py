from lumo.contrib import pickle
import os


def test_dump():
    fn = 'a.aa'
    try:
        with pickle.cached(fn) as cfn:
            pickle.dump(fn, cfn)
            assert os.path.exists(cfn)
            assert not os.path.exists(fn)

        assert not os.path.exists(cfn)
        assert os.path.exists(fn)

        assert (pickle.load(fn) == fn)

        assert not os.path.exists(fn)
    except BaseException as e:
        raise
    finally:
        if os.path.exists(fn):
            os.remove(fn)
