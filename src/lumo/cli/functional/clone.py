import os.path

import git
from git import RemoteProgress
from urllib3.util import parse_url
from lumo import Logger

log = Logger()
template_map = {
    'classify': 'https://github.com/pytorch-lumo/wsl-baselines'
}


def prograss(*args, **kwargs):
    log.inline(*args)


def git_clone(url, alias=None):
    if alias is None:
        u = parse_url(url)
        alias = u.path.split('/')[-1]

    res = git.Repo.clone_from(url, alias, progress=prograss)
    log.newline()
    return res, alias


def git_clone_from_template(template, alias=None):
    url = template_map[template]
    return git_clone(url, alias)
