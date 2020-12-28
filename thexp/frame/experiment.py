import atexit
import json
import os
import re
import sys
import warnings
from collections import namedtuple

from thexp.globals import _CONFIGL, _GITKEY, _FNAME, _OS_ENV, _PLUGIN_KEY, _BUILTIN_PLUGIN, _PLUGIN_DIRNAME
from thexp.utils.dates import curent_date

from .configs import globs

test_dir_pt = re.compile("[0-9]{4}\.[a-z0-9]{7}")
exception = namedtuple('exception_status', ['exc_type', 'exc_value', 'exc_tb'])


class Experiment:
    """
    用于目录管理和常量配置管理
    """
    user_level = _CONFIGL.running
    exp_level = _CONFIGL.globals
    repo_level = _CONFIGL.repository
    count = 0
    whole_tests = []

    def __init__(self, exp_name):
        self._start_time = None
        self._time_fmt = "%Y-%m-%d %H:%M:%S"

        self._exp_name = exp_name
        self._exp_dir = None
        self._test_dir = None
        self._project_dir = None

        self._repo = None
        self._commit = None

        self._end_state = False  # 是否试验已结束
        self._exc_dict = None  # type:exception

        self._config = globs

        self._hold_dirs = []
        self._tags = {}
        self._plugins = {}

        self._in_main = True
        self._initial()

    def __getstate__(self):
        res = {
            '_start_time': self._start_time,
            '_time_fmt': self._time_fmt,

            '_exp_name': self._exp_name,
            '_exp_dir': self.exp_dir,
            '_test_dir': self.test_dir,
            '_project_dir': self.project_dir,

            '_hold_dirs': self._hold_dirs,
            '_tags': self._tags,
            '_plugins': self._plugins,

            '_commit': self.commit.hexsha,

            '_exc_dict': self._exc_dict,
            '_end_state': self._end_state,

            '_in_main': self._in_main,
        }
        emp_ks = [k for k in res if res[k] is None]
        for k in emp_ks:
            res.pop(k)
        return res

    def __setstate__(self, state: dict):
        from gitdb.util import hex_to_bin
        from git import Commit
        self._start_time = state['_start_time']
        self._time_fmt = state['_time_fmt']

        self._exp_name = state['_exp_name']
        self._exp_dir = state['_exp_dir']
        self._test_dir = state['_test_dir']
        self._project_dir = state['_project_dir']

        self._hold_dirs = state['_hold_dirs']
        self._plugins = state['_plugins']
        self._tags = state['_tags']

        self._exc_dict = state.get('_exc_dict', None)
        self._end_state = state['_end_state']

        self._in_main = state['_in_main']

        self._repo = None
        self._commit = Commit(self.repo, hex_to_bin(state['_commit']))
        self._config = globs

    def disable_write(self):
        self._in_main = False

    def __getitem__(self, item):
        return self._config[item]

    def __setitem__(self, key, value):
        self._config[key] = value

    def _initial(self):
        """
        初始化，会在实例被创建完成后调用

        操作：
            注册 系统退出 回调
            注册 repository
            注册 实验
            提交
        """
        from thexp.utils.repository import commit as git_commit
        if self._start_time is None:
            self._start_time = curent_date(self._time_fmt)
            sys.excepthook = self.exc_end
            atexit.register(self.end)

            self._regist_repo()
            self._regist_exp()

            # 目录下所有thexp项目
            if _OS_ENV.THEXP_COMMIT_DISABLE not in os.environ:
                git_commit(self.repo, _GITKEY.commit_key, branch_name=_GITKEY.thexp_branch)

            self._write()
        else:
            warnings.warn("start the same experiment twice is not suggested.")

    def _regist_repo(self):
        """
        在相应位置注册当前实验保存的项目级别的路径（project_dir）与该repo的相应位置，便于回溯。

        共在三个位置添加相应关系（每次都会确认，未添加会添加，添加则覆盖）

        :param exp_repo:  Exprepo对象
        :return:
        """

        # 在全局添加 项目下实验存储路径与相应repo的关系，由此可以存储所有的实验存储目录
        from thexp.utils.paths import home_dir
        global_repofn = os.path.join(home_dir(), _FNAME.repo)
        if os.path.exists(global_repofn):
            with open(global_repofn, 'r', encoding='utf-8') as r:
                res = json.load(r)
        else:
            res = {}
        res[self.project_dir] = self.repo.working_dir
        with open(global_repofn, 'w', encoding='utf-8') as w:
            json.dump(res, w)

        # 在项目级别的实验存储目录下写下相应 repo ，用于方便获取
        local_repofn = os.path.join(self.project_dir, _FNAME.repopath)
        with open(local_repofn, 'w', encoding='utf-8') as w:
            w.write(self.repo.working_dir)

        # 在repo目录下记录所用过的实验存储路径到 .expsdirs，方便回溯
        repo_expdirfn = os.path.join(self.repo.working_dir, _FNAME.expsdirs)
        if os.path.exists(repo_expdirfn):
            with open(repo_expdirfn, 'r', encoding='utf-8') as r:
                _expsdirs = [i.strip() for i in r.readlines()]
        else:
            _expsdirs = []

        if self.expsdir not in _expsdirs:
            _expsdirs.append(self.expsdir)
            with open(repo_expdirfn, 'w', encoding='utf-8') as w:
                w.write("\n".join(_expsdirs))

    def _regist_exp(self):
        """
        在实验目录下存储该实验对应的repo

        Returns:
            返回是否修改，如有修改，可能需要重新提交保证最新
        """
        # 在 exp_dir 下存储对应的 repo 路径
        local_repofn = os.path.join(self.exp_dir, _FNAME.repopath)
        with open(local_repofn, 'w', encoding='utf-8') as w:
            w.write(self.repo.working_dir)

    def _write(self, **extra):
        """将试验状态写入试验目录中的 info.json """
        if self._end_state:
            return
        if not self._in_main:
            return
        res = dict(
            repo=self.repo.working_dir,
            argv=sys.argv,
            exp_name=self._exp_name,
            exp_dir=self.exp_dir,
            test_name=os.path.basename(self.test_dir),
            test_dir=self.test_dir,
            root_dir=self.expsdir,
            project_name=self.project_name,
            project_iname=os.path.basename(self.project_dir),
            project_dir=self.project_dir,
            commit_hash=self.commit_hash,
            short_hash=self.test_hash,
            dirs=self._hold_dirs,
            time_fmt=self._time_fmt,
            start_time=self._start_time,
            tags=self._tags,
            plugins=self._plugins,
            **extra,
        )
        with open(self.test_info_fn, 'w', encoding='utf-8') as w:
            json.dump(res, w, indent=2)

    @property
    def repo(self):
        from ..utils.repository import load_repo
        if self._repo is None:
            self._repo = load_repo()
        return self._repo

    @property
    def commit(self):
        from thexp.utils.repository import commit as git_commit
        if self._commit is None:
            self._commit = git_commit(self.repo, _GITKEY.commit_key, branch_name=_GITKEY.thexp_branch)
        return self._commit

    @property
    def test_hash(self) -> str:
        if self.commit is None:
            return ""

        return self.commit.hexsha[:8]

    @property
    def test_name(self) -> str:
        return os.path.basename(self.test_dir)

    @property
    def commit_hash(self) -> str:
        if self.commit is None:
            return ""
        return self.commit.hexsha

    @property
    def project_name(self) -> str:
        return self[_GITKEY.projname]

    @property
    def projkey(self) -> str:
        # return self[_CONFIGL]
        return '{}.{}'.format(self.project_name, self[_GITKEY.uuid])

    @property
    def expsdir(self) -> str:
        """
        配置中存储的，可以用于所有项目实验日志记录的根路径
        Returns:

        """
        return self[_GITKEY.expsdir].rstrip('\\/')

    @property
    def project_dir(self) -> str:
        """
        实验根目录下，项目级别的目录
        Returns:

        """
        if self._project_dir is None:
            project_dir_ = self.projkey
            self._project_dir = os.path.join(self.expsdir, project_dir_)
            os.makedirs(self._project_dir, exist_ok=True)

        return self._project_dir

    @property
    def exp_dir(self) -> str:
        """
        在每个项目下的每个实验的目录
        Returns:

        """
        if self._exp_dir is None:
            self._exp_dir = os.path.join(self.project_dir, self._exp_name)
            os.makedirs(self._exp_dir, exist_ok=True)
            with open(os.path.join(self._exp_dir, _FNAME.repopath), 'w', encoding='utf-8') as w:
                w.write(self.repo.working_dir)
        return self._exp_dir

    @property
    def test_dir(self) -> str:
        """
        获取当前 exp 目录下的 test_dir
        命名方式： 通过 序号.hash 的方式命名每一次实验
        会进行一系列判断保证在硬盘中的任何位置不会出现同名试验
        其中，hash 保证在不同时间运行的试验文件名不同，序号保证在同一进程运行的多次试验文件名不同
        {:04d}.{hash}

        Returns: 一个全局唯一的test_dir，(绝对路径)

        Notes:
            命名时还会确保存储路径下不会存在相同的序号（在生成时如果有相同序号则一直+1），方法为
            第一次获取试验目录时记录生成的序号，之后所有的序号都在此基础上生成，但如果 exp 目录变更，则
        """
        if self._test_dir is None:
            fs = os.listdir(self.exp_dir)

            if Experiment.count != 0:  # 如果
                i = self.count + 1
            else:
                atexit.register(final_report)
                i = len([f for f in fs if re.search(test_dir_pt, f) is not None]) + 1

            cf_set = {f.split('.')[0] for f in fs}  # 首位元素永远存在，不需要判断其他文件
            while "{:04d}".format(i) in cf_set:
                i += 1

            self._test_dir = os.path.join(self.exp_dir, "{:04d}.{}".format(i, self.test_hash))
            os.makedirs(self._test_dir, exist_ok=True)
            Experiment.count = i
            Experiment.whole_tests.append(os.path.basename(self._test_dir))

        return self._test_dir

    @property
    def test_info_fn(self) -> str:
        return os.path.join(self.test_dir, _FNAME.info)

    @property
    def plugins(self) -> dict:
        return self._plugins

    def makedir(self, name):
        """
        创建 test 级别的目录
        :param name: 目录名
        :return:  返回创建目录的绝对路径
        """
        d = os.path.join(self.test_dir, name)
        os.makedirs(d, exist_ok=True)
        self._hold_dirs.append(name)
        return d

    def make_exp_dir(self, name) -> str:
        """创建 exp 级别的目录"""
        d = os.path.join(self.exp_dir, name)
        os.makedirs(d, exist_ok=True)
        return d

    def end(self, end_code=0, **extra):
        """
        手动结束试验时调用该方法可以传入其他信息写入文件，
        该方法同时还通过 atexit 注册了退出钩子，如果不调用该方法，则在程序正常退出前会自动调用该方法写入结束文件。
        通过pycharm等IDE结束时，由于其结束机制不是通过抛出 KeyboardInterrupt 异常停止的，所以该方法存在没有正常调
        用的情况，此时文件中不会记录结束用时和退出状态码，也可以做为实验失败的判断条件。

        注意，结束状态仅会调用一次，无论是正常结束（对应方法end()）还是异常结束（对应方法exc_end()），
        在调用后的任何对实验的操作均不会写入到文件中
        :param end_code: 退出状态码，表明程序是否正常结束，以免引起数据混乱
        :param extra:
        :return:
        """
        self._write(
            end_time=curent_date(self._time_fmt),
            end_code=end_code,
            **extra
        )
        self._end_state = True

    def exc_end(self, exc_type, exc_val, exc_tb):
        """
        该方法注册了异常钩子，无需手动调用，在出现异常且没有处理导致程序退出的情况时，会通过该方法记录程序失败原因，
        包括异常类型和提示，同时会将异常栈内容输出到试验目录下的 Exception 文件中
        :param exc_type:  异常传入类型
        :param exc_val:  异常传入示例
        :param exc_tb:  traceback
        :return:
        """
        import traceback
        with open(os.path.join(self.test_dir, _FNAME.Exception), 'w', encoding='utf-8') as w:
            w.write("".join(traceback.format_exception(exc_type, exc_val, exc_tb)))
        self.end(
            end_code=1,
            exc_type=traceback.format_exception_only(exc_type, exc_val)[-1].strip()
        )
        self._exc_dict = exception(exc_tb, exc_val, exc_tb)
        self._end_state = True
        traceback.print_exception(exc_type, exc_val, exc_tb)

    def add_tag(self, name, *plugins):
        """
        插件的插件，用来表示tag对应的某个插件利用另一个插件做了某些什么事情
        :param name:
        :param plugins:
        :return:
        """
        self._tags[name] = plugins
        self._write()

    def add_plugin(self, key: str, value: dict = None):
        """
        注册试验中使用的插件。


        Args:
            key: 插件名
            value: 若为空，则为空字典

        Returns:

        """
        if value is None:
            value = {}
        self._plugins[key] = value
        self._write()

    def add_params(self, params):
        """
        ```python
        from thexp import Params
        p = Params()
        exp.add_params(p)
        ```
        """
        kwargs = {
            _PLUGIN_KEY.PARAMS.param_hash: params.hash(),
        }
        self.add_plugin(_BUILTIN_PLUGIN.params, kwargs)
        params.to_json(os.path.join(self.test_dir, _FNAME.params))

    def add_logger(self, fn: str):
        """
        Examples:

        ```python
        from thexp import Logger
        logger = Logger()
        fn = logger.add_log_dir(self.experiment.test_dir)
        exp.add_logger(fn)
        ```
        """
        kwargs = {
            _PLUGIN_KEY.LOGGER.log_dir: self.test_dir,
            _PLUGIN_KEY.LOGGER.fn: fn,
        }
        self.add_plugin(_BUILTIN_PLUGIN.logger, kwargs)

    def add_board(self):
        """
        Examples:
        ```python
        from torch.utils.tensorboard import SummaryWriter

        kwargs = exp.add_board()
        res = SummaryWriter(**kwargs)
        ```
        """
        d = self.makedir(_PLUGIN_DIRNAME.writer)
        kwargs = {
            _PLUGIN_KEY.WRITER.log_dir: d,
            _PLUGIN_KEY.WRITER.filename_suffix: '.bd'
        }
        self.add_plugin(_BUILTIN_PLUGIN.writer, kwargs)
        return kwargs

    def add_saver(self):
        """
        ```python
        from thexp import Saver
        kwargs = self.experiment.add_saver()
        Saver(**kwargs)
        ```
        """
        d = self.makedir(_PLUGIN_DIRNAME.saver)

        kwargs = {
            _PLUGIN_KEY.SAVER.max_to_keep: 3,
            _PLUGIN_KEY.SAVER.ckpt_dir: d
        }
        self.add_plugin(_BUILTIN_PLUGIN.saver, kwargs)
        return kwargs

    def add_rndmanager(self):
        """
        ```python
        from thexp import RndManager
        kwargs = exp.add_rndmanager()
        return RndManager(**kwargs)
        ```
        """
        d = self.make_exp_dir(_PLUGIN_DIRNAME.rnd)

        kwargs = {
            _PLUGIN_KEY.RND.save_dir: d
        }
        self.add_plugin(_BUILTIN_PLUGIN.rnd, kwargs)
        return kwargs

    def config_items(self):
        """
        三个等级的配置文件的内容
        :return:
        """
        return self._config.items()

    def add_config(self, key, value, level=_CONFIGL.globals):

        """
        添加配置
        :param key:
        :param value:
        :param level: 配置级别，查看 globals._CONFIGL
        :return:
        """
        self._config.add_value(key, value, level)

    def regist_exit_hook(self, func):
        """
        提供给用户的额外注册退出钩子，传入的参数有两个，
        第一个参数为该 Experiment 的实例对象
        第二个参数为一个 exception_status 实例，当程序的退出是因为某异常引起时，该参数会被传入
            exception_status 是一个 namedtuple('exception_status', ['exc_type', 'exc_value', 'exc_tb'])
            可以通过 Python 内置的 traceback 库来处理，如

            import traceback
            traceback.print_exception(exc_type, exc_val, exc_tb)
        :param func: 退出前调用的钩子，
        :return:
        """
        import atexit
        def exp_func():
            func(self, self._exc_dict)

        atexit.register(exp_func)


def final_report():
    print('Test end:')
    import pprint
    pprint.pprint(Experiment.whole_tests)
