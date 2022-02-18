import inspect
from os import path
from Finetuning import fine_tuner
import os
import sys


def read_file(filename, depth=1):
    """reads the file"""

    if not filename:
        raise ValueError("filename not supplied")

    # Expanding filename
    filename = os.path.expanduser(filename)
    file_path = os.path.join(
        os.path.dirname(inspect.getfile(sys._getframe(depth))), filename
    )
    print(inspect.getsource(sys._getframe(depth)))

    if not os.path.isfile(file_path):

        raise ValueError("file {} not found".format(filename))

    with open(file_path, "r") as data:
        return data.read()


f = read_file(filename="./LM_DEMO.py")
print(type(f))