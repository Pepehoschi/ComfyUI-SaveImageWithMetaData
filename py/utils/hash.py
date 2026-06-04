import hashlib
import os


cache_model_hash = {}


def calc_hash(filename):
    if filename is None:
        return None

    try:
        filename = os.fspath(filename)
    except TypeError:
        return None

    if not os.path.isfile(filename):
        return None

    if filename in cache_model_hash:
        return cache_model_hash[filename]
    sha256_hash = hashlib.sha256()

    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    model_hash = sha256_hash.hexdigest()[:10]

    cache_model_hash[filename] = model_hash
    return model_hash
