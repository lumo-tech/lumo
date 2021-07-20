"""
initialize directory, create config file, including
 - ~/.lumo/
 - ~/.cache/lumo/
"""
import os


def initialize():
    from lumo.utils.template_creator import Template
    from lumo.proc.path import libhome
    if os.path.exists(libhome()):
        return False

    (
        Template()
            .add_directory('experiments')
            .add_file('config.json', info='')
            .create(libhome())
    )

    return True
