# TODO colored text
import ctypes
import sys

if sys.platform == "win32" and sys.stdin.isatty():
    FOREGROUND_BLACK = 0x00  # black.
    FOREGROUND_DARKBLUE = 0x01  # dark blue.
    FOREGROUND_DARKGREEN = 0x02  # dark green.
    FOREGROUND_DARKSKYBLUE = 0x03  # dark skyblue.
    FOREGROUND_DARKRED = 0x04  # dark red.
    FOREGROUND_DARKPINK = 0x05  # dark pink.
    FOREGROUND_DARKYELLOW = 0x06  # dark yellow.
    FOREGROUND_DARKWHITE = 0x07  # dark white.
    FOREGROUND_DARKGRAY = 0x08  # dark gray.
    FOREGROUND_BLUE = 0x09  # blue.
    FOREGROUND_GREEN = 0x0a  # green.
    FOREGROUND_SKYBLUE = 0x0b  # skyblue.
    FOREGROUND_RED = 0x0c  # red.
    FOREGROUND_PINK = 0x0d  # pink.
    FOREGROUND_YELLOW = 0x0e  # yellow.
    FOREGROUND_WHITE = 0x0f  # white.
    FOREGROUND_PURPLE = FOREGROUND_PINK
    FOREGROUND_CYAN = FOREGROUND_PINK

    BACKGROUND_BLACK = 0x00
    BACKGROUND_DARKBLUE = 0x10  # dark blue.
    BACKGROUND_DARKGREEN = 0x20  # dark green.
    BACKGROUND_DARKSKYBLUE = 0x30  # dark skyblue.
    BACKGROUND_DARKRED = 0x40  # dark red.
    BACKGROUND_DARKPINK = 0x50  # dark pink.
    BACKGROUND_DARKYELLOW = 0x60  # dark yellow.
    BACKGROUND_DARKWHITE = 0x70  # dark white.
    BACKGROUND_DARKGRAY = 0x80  # dark gray.
    BACKGROUND_BLUE = 0x90  # blue.
    BACKGROUND_GREEN = 0xa0  # green.
    BACKGROUND_SKYBLUE = 0xb0  # skyblue.
    BACKGROUND_RED = 0xc0  # red.
    BACKGROUND_PINK = 0xd0  # pink.
    BACKGROUND_YELLOW = 0xe0  # yellow.
    BACKGROUND_WHITE = 0xf0  # white.
    BACKGROUND_PURPLE = BACKGROUND_PINK
    BACKGROUND_CYAN = BACKGROUND_PINK
else:
    FOREGROUND_BLACK = 30
    FOREGROUND_RED = 31
    FOREGROUND_GREEN = 32
    FOREGROUND_YELLOW = 33
    FOREGROUND_BLUE = 34
    FOREGROUND_PURPLE = 35
    FOREGROUND_CYAN = 36
    FOREGROUND_WHITE = 37
    FOREGROUND_SKYBLUE = FOREGROUND_BLUE
    FOREGROUND_PINK = FOREGROUND_PURPLE
    FOREGROUND_DARKBLUE = FOREGROUND_BLUE
    FOREGROUND_DARKGREEN = FOREGROUND_GREEN
    FOREGROUND_DARKSKYBLUE = FOREGROUND_BLUE
    FOREGROUND_DARKRED = FOREGROUND_RED
    FOREGROUND_DARKPINK = FOREGROUND_CYAN
    FOREGROUND_DARKYELLOW = FOREGROUND_YELLOW
    FOREGROUND_DARKWHITE = FOREGROUND_WHITE
    FOREGROUND_DARKGRAY = FOREGROUND_BLACK

    BACKGROUND_BLACK = 40
    BACKGROUND_RED = 41
    BACKGROUND_GREEN = 42
    BACKGROUND_YELLOW = 43
    BACKGROUND_BLUE = 44
    BACKGROUND_PURPLE = 45
    BACKGROUND_CYAN = 46
    BACKGROUND_WHITE = 47
    BACKGROUND_SKYBLUE = BACKGROUND_BLUE
    BACKGROUND_PINK = BACKGROUND_PURPLE
    BACKGROUND_DARKBLUE = BACKGROUND_BLUE
    BACKGROUND_DARKGREEN = BACKGROUND_GREEN
    BACKGROUND_DARKSKYBLUE = BACKGROUND_BLUE
    BACKGROUND_DARKRED = BACKGROUND_RED
    BACKGROUND_DARKPINK = BACKGROUND_PURPLE
    BACKGROUND_DARKYELLOW = BACKGROUND_YELLOW
    BACKGROUND_DARKWHITE = BACKGROUND_WHITE
    BACKGROUND_DARKGRAY = BACKGROUND_BLACK

MODE_NORMAL = 0
MODE_BOLD = 1
MODE_UNDERLINE = 4
MODE_BLINK = 5
MODE_INVERT = 7
MODE_HIDE = 8

if sys.platform == "win32" and sys.stdin.isatty():
    STD_INPUT_HANDLE = -10
    STD_OUTPUT_HANDLE = -11
    STD_ERROR_HANDLE = -12
    std_out_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    std_err_handle = ctypes.windll.kernel32.GetStdHandle(STD_ERROR_HANDLE)
    # _handle_dict = dict(
    #     stdout=std_out_handle,
    #     stderr=std_err_handle
    # )
else:
    std_out_handle = sys.stdout
    std_err_handle = sys.stderr
_handle_dict = dict(
    stdout=sys.stdout,
    stderr=sys.stderr
)


def set_cmd_text_color(color, handle=std_out_handle):
    Bool = ctypes.windll.kernel32.SetConsoleTextAttribute(handle, color)
    return Bool


def reset_color():
    set_cmd_text_color(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE)


def in_terminal():
    '''
    在win环境中用python xx.py命令时，只支持获取句柄更改的手段，其他时候，可以用 \033
    :return:
    '''
    return sys.stdin.isatty()


def in_win():
    return sys.platform == "win32"


def cprint(*args, fore=None, back=None, sep=" ", end="", handle="stdout", flush=True):
    value = sep.join(*args)

    if fore is None and back is None:
        reset_color()
    elif back is None:
        set_cmd_text_color(fore)
    elif fore is None:
        set_cmd_text_color(back)
    else:
        set_cmd_text_color(fore | back)

    print(value, end=end, file=_handle_dict[handle], flush=flush)
    # sys.stdout.write(str(value))
    # if flush:
    #     sys.stdout.flush()

    reset_color()
    return cprint


# def eprint(*args, fore=None, back=None, mode=None, sep=" ", end="", handle="stdout", flush=True):
#     value = sep.join(*args)
#
#     if mode is None:
#         mode = MODE_NORMAL
#     if fore is None and back is None:
#         print(f"\033[{mode}m{value}\033[0m", end=end, file=_handle_dict[handle], flush=flush)
#     elif back is None:
#         print(f"\033[{mode};{fore}m{value}\033[0m", end=end, file=_handle_dict[handle], flush=flush)
#     elif fore is None:
#         print(f"\033[{mode};;{back}m{value}\033[0m", end=end, file=_handle_dict[handle], flush=flush)
#     else:
#         print(f"\033[{mode};{fore};{back}m{value}\033[0m", end=end, file=_handle_dict[handle], flush=flush)


def uprint(*args,
           fore=None,
           back=None,
           mode=None,
           sep=" ",
           end="",
           handle="stdout",
           flush=True):
    '''
    Prints the colored values to sys.stdout or sys.stderr.
    :param fore:
    :param back:
    :param mode:
    :param sep: string inserted between values, default a space.
    :param end: string appended after the last value, default a newline.
    :param handle: str, "stdout" or "stderr"
        note:the stderr haven't be tested.
    :param flush: whether to forcibly flush the stream.
    :param continuous: default True, if True, the return value will be a lambda method wrapped the uprint mathod
                and the options.
    :return: uprint method, you can use
        uprint()()()() to output strings with diff color.
    '''
    args = [str(i) for i in args]
    if in_win() and in_terminal():
        cprint(args, fore=fore, back=back, sep=sep, end=end, handle=handle, flush=flush)
    else:
        pass
        # eprint(args, fore=fore, back=back, mode=mode, sep=sep, end=end, handle=handle, flush=flush)

    def extendprint(*args,
                    fore=fore,
                    back=back,
                    mode=mode,
                    sep=sep,
                    end=end,
                    handle=handle,
                    flush=flush):
        return uprint(*args, fore=fore, back=back, mode=mode, sep=sep, end=end, handle=handle, flush=flush)

    return extendprint
