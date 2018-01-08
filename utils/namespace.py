import sys
from types import SimpleNamespace

class Namespace(SimpleNamespace):

    def __init__(self, *args, **kwargs):
        super(Namespace, self).__init__(**kwargs)
        if args is not None:
            self.__dict__.update(*args)

    def __len__(self):
        return len(self.__dict__)

    def __contains__(self, item):
        return item in self.__dict__

    def get(self, key, default_value=None):
        if not key in self.__dict__:
            self.__dict__[key] = default_value

        return self.__dict__.get(key)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __getitem__(self, key):
        if not key in self.__dict__:
            self.__dict__[key] = None

        return self.__dict__.get(key)

    def _collect_props(self,  props, iterable, name=''):
        
        for key, value in iterable:
            if isinstance(value, (Namespace, dict)):
                props = self._collect_props(props, value.items(), key)
            else:
                if name != '':
                    val = [name, key, value]
                else:
                    val = [key, value]
                props.append(val)
        return props

    def summary(self, f=sys.stdout):

        props = self._collect_props([], self.items())

        for prop in sorted(props, key=lambda prop: '{0} {1}'.format(prop[0], prop[1])):
            if len(prop) == 3:
                print(prop[0], prop[1], prop[2], file=f)
            else:
                print(prop[0], prop[1], file=f)