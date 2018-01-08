import re
from contextlib import contextmanager
from time import time
from datetime import datetime
from collections import OrderedDict
import inspect

import numpy as np

_patterns = {'private' : re.compile(r"^_[^_].*$"),
             'public'  : re.compile(r"^[^_].*$"),
             'all'     : re.compile(r"^(?!__).*$")}
def is_target_method(name, method, scope, target):
    if target is not None:
        return name == target
    return (callable(method) or
           (hasattr(method, '__func__') and isinstance(method.__func__, (RuntimeMeaser, Tracer)))) \
            and _patterns[scope].match(name)
_measers = {}

def set_measer(observable, scope='all', target=None):
        methods = OrderedDict()
        obs_name = get_class_name(observable)
        for name, method in inspect.getmembers(observable, predicate=inspect.ismethod):
            
            if is_target_method(name, method, scope, target):
                wraped_method = RuntimeMeaser(obs_name, method)
                methods[name] = wraped_method
                setattr(observable, name, wraped_method)

        global _measers
        _measers[observable] = methods

        return observable

def measers_summary():
    for obverbable, methods in _measers.items():
        print('====== {} ======'.format(type(obverbable)))
        for name, measer in methods.items():
            if measer.count > 0:
                print('{}: count:{} total_time:{:0.5f}'.format(name, measer.count, measer.time))

class RuntimeMeaser(object):
    def __init__(self, obj_name, method):
        self.obj_name = obj_name
        self.method = method
        self.time = 0
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.current_run_time = 0
        start_t = time()
        r = self.method(*args, **kwargs)
        self.count += 1
        self.current_run_time = time() - start_t 
        self.time += self.current_run_time
        return r

_tracers = {}
def set_tracer(traceable, scope='all', target=None):
    methods = {}
    obj_name = get_class_name(traceable)
    for name, method in inspect.getmembers(traceable):
        if is_target_method(name, method, scope, target):
            wraped_method = Tracer(obj_name, name, method)
            methods[name] = wraped_method
            print('set_tracer',name, target)
            setattr(traceable, name, wraped_method)
    global _tracers
    _tracers[traceable] = methods

    return traceable

_trace = []
class Tracer(object):
    def __init__(self, obj_name, name, method):
        self.obj_name = obj_name
        self.name = name
        self.method = method
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        _trace.append((True, self.obj_name, self.name, self.count, datetime.now(), args, kwargs))
        r = self.method(*args, **kwargs)
        run_time = getattr(self.method, 'current_run_time', None)
        _trace.append((False, self.obj_name, self.name, self.count, datetime.now(), run_time, r))
        return r

def tracers_summary():
    print('====== trace start ======')
    for i, trace_info in enumerate(_trace):
        
        if trace_info[0]:
            _, obj_name, name, count, dt, args, kwargs = trace_info
            print('[{}] {}.{} ({}) start:{} *args{}, **kwargs{}'.format(
                i,
                obj_name,
                name,
                count,
                dt.strftime('%Y-%m-%d %H:%M:%S'),
                get_args(args),
                get_kwargs(kwargs)))
        else:
            _, obj_name, name, count, dt, run_time, _ = trace_info
            print('[{}] {}.{} ({}) end  :{}{}'.format(
                i,
                obj_name,
                name,
                count,
                dt.strftime('%Y-%m-%d %H:%M:%S'),
                ' ({:0.5f})'.format(run_time) if run_time is not None else \
                ''))

def get_class_name(obj):
    t = type(obj)
    return re.split(r"'", str(t))[1]

def get_arg(arg):
    if np.isscalar(arg):
        return arg
    elif isinstance(arg, (list, dict, tuple, np.ndarray)):
        return dict(clazz=get_class_name(arg), len=len(arg))


def get_args(args):
    if np.isscalar(args):
        return args
    return tuple([get_arg(arg) for arg in args])

def get_kwargs(kwargs):
    ret_args = {}
    for k, v in kwargs.items():
        ret_args[k] = get_arg(v)
                
    return ret_args
