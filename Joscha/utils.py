import os


def list_dir(path):
    files = os.listdir(path)
    # filter out hidden files
    files = [f for f in files if not f.startswith('.')]
    return files