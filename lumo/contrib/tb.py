import sys

from absl import app
from tensorboard import default
from tensorboard import main_lib
from tensorboard import program
from tensorboard.plugins import base_plugin
from tensorboard.uploader import uploader_subcommand
from tensorboard.util import tb_logging

logger = tb_logging.get_logger()

from typing import overload


@overload
def run_board(logdir, host=None, port=None): pass


def run_board(logdir, **kwargs):
    main_lib.global_init()

    tensorboard = program.TensorBoard(
        plugins=default.get_plugins(),
        subcommands=[uploader_subcommand.UploaderSubcommand()],
    )
    try:
        app.run(tensorboard.main,
                argv=[
                    sys.argv[0],
                    '--logdir', logdir,
                ],
                flags_parser=tensorboard.configure)
    except base_plugin.FlagsError as e:
        print("Error: %s" % e, file=sys.stderr)
        sys.exit(1)
