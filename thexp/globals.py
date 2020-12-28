"""
define some global string variables in case name mistakes
"""
from thexp import __VERSION__


class _REPOJ:
    repopath = 'repopath'
    exps = 'exps'
    exp_root = 'exp_root'


class _INFOJ:
    repo = 'repo'
    argv = 'argv'
    test_name = 'test_name'
    commit_hash = 'commit_hash'
    short_hash = 'short_hash'
    dirs = 'dirs'
    time_fmt = 'time_fmt'
    start_time = 'start_time'
    tags = 'tags'
    _tags = '_tags'
    _ = '_'
    plugins = 'plugins'
    end_time = 'end_time'
    end_code = 'end_code'


class _CONFIGL:
    running = 'running'
    globals = 'globals'
    repository = 'repository'


class _GITKEY:
    thexp = 'thexp'
    projname = 'projname'
    expsdir = 'expsdir'
    uuid = 'uuid'
    section_name = 'thexp'
    thexp_branch = 'experiment'
    commit_key = 'thexp-commit'


class _FNAME:
    Exception = 'Exception'
    info = 'info.v1.json'
    repo = 'repo.v1.json'
    params = 'params.v1.json'
    repopath = '.repopath'
    expsdirs = '.expsdirs'
    gitignore = ".gitignore"
    gitignore_version = '.thexp.{}'.format(__VERSION__)


class _TEST_BUILTIN_STATE:
    hide = 'hide'
    fav = 'fav'


class _ML:
    train = 'train'
    test = 'test'
    eval = 'eval'
    cuda = 'cuda'


class _BUILTIN_PLUGIN:
    trainer = 'trainer'
    params = 'params'
    writer = 'writer'
    logger = 'logger'
    saver = 'saver'
    rnd = 'rnd'


class _PLUGIN_DIRNAME:
    writer = 'board'
    writer_tmp = 'board_tmp'
    saver = 'modules'
    rnd = 'rnd'


class _PLUGIN_KEY:
    class WRITER:
        log_dir = 'log_dir'
        filename_suffix = 'filename_suffix'
        dir_name = 'board'

    class LOGGER:
        log_dir = 'log_dir'
        fn = 'fn'

    class RND:
        save_dir = 'save_dir'

    class PARAMS:
        param_hash = 'param_hash'

    class TRAINER:
        path = 'path'
        doc = 'doc'
        fn = 'module'
        class_name = 'class_name'

    class SAVER:
        max_to_keep = 'max_to_keep'
        ckpt_dir = 'ckpt_dir'


class _INDENT:
    tab = '  '
    ttab = '    '
    tttab = '      '


class _DLEVEL:
    proj = 'proj'
    exp = 'exp'
    test = 'test'


class _OS_ENV:
    CUDA_VISIBLE_DEVICES = 'CUDA_VISIBLE_DEVICES'
    THEXP_COMMIT_DISABLE = 'THEXP_COMMIT_DISABLE'
    IGNORE_REPO = 'THEXP_IGNORE_REPO'
