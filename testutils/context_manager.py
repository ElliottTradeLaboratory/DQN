import sys
import re
import contextlib

class OutputGetter(object):
    def __init__(self, stream):
        self.stream = stream
        self.outputs = []
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        if not re.match('^$', data):
            self.outputs.append(data)
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
        if not re.match('^$', data):
            self.outputs.append(data)
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

@contextlib.contextmanager
def get_stdout():

    org_stdout = sys.stdout
    stdout = OutputGetter(sys.stdout)
    sys.stdout = stdout
    try:
        yield stdout
    finally:
        sys.stdout = org_stdout
