from lumo.proc.dist import *


def test_local_rank(monkeypatch):
    monkeypatch.setenv('LOCAL_RANK', '0')
    assert local_rank() == 0

    monkeypatch.setenv('LOCAL_RANK', '1')
    assert local_rank() == 1


def test_world_size(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '4')
    assert world_size() == 4


def test_is_dist(monkeypatch):
    monkeypatch.setenv('LOCAL_RANK', '0')
    assert is_dist() == True


def test_is_main(monkeypatch):
    monkeypatch.setenv('LOCAL_RANK', '0')
    assert is_main() == True
