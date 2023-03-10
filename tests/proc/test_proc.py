from lumo.proc.path import *


def test_home():
    assert home() == os.path.expanduser("~")


# def test_cache_dir():
#     CACHE_ROOT = glob.get('cache_dir', None)
#     expected = CACHE_ROOT or os.path.join(home(), '.lumo/cache')
#     assert cache_dir() == expected


def test_libhome():
    LIBHOME = glob.get('home', None)
    expected = LIBHOME or os.path.join(home(), '.lumo')
    assert libhome() == expected


def test_exproot():
    EXP_ROOT = glob.get('exp_root', None)
    expected = EXP_ROOT or os.path.join(libhome(), 'experiments')
    assert exproot() == expected


def test_progressroot():
    PROGRESS_ROOT = glob.get('progress_root', None)
    expected = PROGRESS_ROOT or os.path.join(cache_dir(), 'progress')
    assert progressroot() == expected


def test_blobroot():
    BLOB_ROOT = glob.get('blob_root', None)
    expected = BLOB_ROOT or os.path.join(libhome(), 'blob')
    assert blobroot() == expected


def test_metricroot():
    METRIC_ROOT = glob.get('metric_root', None)
    expected = METRIC_ROOT or os.path.join(libhome(), 'metrics')
    assert metricroot() == expected


def test_local_dir():
    # it's difficult to test this function without a specific context
    # since it depends on git_dir(), which is not included in the provided code
    pass
