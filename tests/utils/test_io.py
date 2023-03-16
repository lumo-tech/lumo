import yaml
from lumo.utils.safe_io import *
import os

def test_dump_json(tmpdir):
    # Test that dump_json creates a valid JSON file
    obj = {"a": 1, "b": 2}
    fn = os.path.join(str(tmpdir), "test.json")
    dump_json(obj, fn)
    with open(fn, "r") as f:
        data = json.load(f)
    assert data == obj


def test_dump_yaml(tmpdir):
    # Test that dump_yaml creates a valid YAML file
    obj = {"a": 1, "b": 2}
    fn = os.path.join(str(tmpdir), "test.yaml")
    dump_yaml(obj, fn)
    with open(fn, "r") as f:
        data = yaml.safe_load(f)
    assert data == obj


def test_dump_state_dict(tmpdir):
    # Test that dump_state_dict creates a valid state dict file
    obj = {"a": torch.randn(3, 3)}
    fn = os.path.join(str(tmpdir), "test.pt")
    dump_state_dict(obj, fn)
    data = torch.load(fn)
    assert (data['a'] == obj['a']).all()


def test_load_json(tmpdir):
    # Test that load_json reads a valid JSON file
    obj = {"a": 1, "b": 2}
    fn = os.path.join(str(tmpdir), "test.json")
    with open(fn, "w") as f:
        json.dump(obj, f)
    data = load_json(fn)
    assert data == obj


def test_load_yaml(tmpdir):
    # Test that load_yaml reads a valid YAML file
    obj = {"a": 1, "b": 2}
    fn = os.path.join(str(tmpdir), "test.yaml")
    with open(fn, "w") as f:
        yaml.safe_dump(obj, f)
    data = load_yaml(fn)
    assert data == obj


def test_load_state_dict(tmpdir):
    # Test that load_state_dict reads a valid state dict file
    obj = {"a": torch.randn(3, 3)}
    fn = os.path.join(str(tmpdir), "test.pt")
    torch.save(obj, fn)
    data = load_state_dict(fn)
    assert (data['a'] == obj['a']).all()


def test_dump_text(tmpdir):
    # Test that dump_text creates a valid text file
    string = "hello\nworld"
    fn = os.path.join(str(tmpdir), "test.txt")
    dump_text(string, fn)
    with open(fn, "r") as f:
        data = f.read()
    assert data == string


def test_load_text(tmpdir):
    # Test that load_text reads a valid text file
    string = "hello\nworld"
    fn = os.path.join(str(tmpdir), "test.txt")
    with open(fn, "w") as f:
        f.write(string)
    data = load_text(fn)
    assert data == string


def test_dump_pkl(tmpdir):
    # Test that dump_pkl creates a valid pickle file
    obj = {"a": 1, "b": 2}
    fn = os.path.join(str(tmpdir), "test.pkl")
    dump_pkl(obj, fn)
    data = load_pkl(fn)
    assert data == obj


def test_cached(tmpdir):
    # Test that cached context manager works correctly
    with cached(os.path.join(str(tmpdir), "test.txt")) as cache_fn:
        # Write some data to the cache file
        with open(cache_fn, "w") as f:
            f.write("hello")
        # Check that the cache file exists
        assert os.path.isfile(cache_fn)
    # Check that the cache file is deleted after exiting the context manager
    assert not os.path.isfile(cache_fn)
