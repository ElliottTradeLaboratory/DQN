import numpy as np

def assert_equal(a, b, msg=None, use_float_equal=False, verbose=0):

    msg = msg + ': ' if msg is not None else ''

    if a is None == b:
        pass

    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        msg = '{}numpy ndarray not equal {} vs {}'.format(msg, np.mean(a), np.mean(b))
        if use_float_equal:
            assert float_equal(a, b, verbose=verbose), msg
        else:
            assert np.all(a == b), msg

    elif (isinstance(a, list) or isinstance(a, tuple)) and \
         (isinstance(b, list) or isinstance(b, tuple)):
         assert len(a) == len(b), '{}length unmatch {} vs {}'.format(msg, len(a), len(b))
         for i, vals in enumerate(zip(a, b)):
            assert_equal(*vals, '{}({}) {}'.format(msg, i, type(b)), use_float_equal=use_float_equal, verbose=verbose)

    elif isinstance(a, dict) and isinstance(b, dict):
         assert len(a) == len(b), '{}length unmatch {} vs {}'.format(msg, a.keys(), b.keys())

         for key, _b in b.items():
            assert key in a
            _a = a[key]
            assert_equal(_a, _b, '{}{}'.format(msg, key), use_float_equal=use_float_equal, verbose=verbose)

    elif (np.isscalar(a) and np.isscalar(b)) or \
          (isinstance(a, type(b)) and hasattr(a, '__eq__')):
        assert a == b, "{}{} '{}' vs '{}'".format(msg, type(b), a, b)

    else:
        assert False, '{}a({}) and b({}), type mismatch, or not comparable because those type has not __eq__.'.format(msg, type(a), type(b))

def assert_not_equal(a, b, msg=None, use_float_equal=False, verbose=0):

    msg = msg + ': ' if msg is not None else ''

    if a is None == b:
        pass

    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        msg = '{}numpy ndarray not equal {} vs {}'.format(msg, np.mean(a), np.mean(b))
        if use_float_equal:
            print('verbose',verbose)
            assert not float_equal(b, a, verbose), msg
        else:
            assert np.any(a != b), msg

    elif (isinstance(a, list) or isinstance(a, tuple)) and \
         (isinstance(b, list) or isinstance(b, tuple)):
         assert len(a) == len(b), '{}length unmatch {} vs {}'.format(msg, len(a), len(b))
         for i, vals in enumerate(zip(a, b)):
            assert_not_equal(*vals, '{}({}) {}'.format(msg, i, type(b)), use_float_equal=use_float_equal, verbose=verbose)

    elif isinstance(a, dict) and isinstance(b, dict):
         assert len(a) == len(b), '{}length unmatch {} vs {}'.format(msg, a.keys(), b.keys())

         for key, _b in b.items():
            assert key in a
            _a = a[key]
            assert_not_equal(_a, _b, '{}{}'.format(msg, key), use_float_equal=use_float_equal, verbose=verbose)

    elif (np.isscalar(a) and np.isscalar(b)) or \
          (isinstance(a, type(b)) and hasattr(a, '__eq__')):
        assert a != b, '{}{} {} vs {}'.format(msg, type(b), a, b)

    else:
        assert False, '{}a({}) and b({}), type mismatch, or not comparable because those type has not __eq__.'.format(msg, type(a), type(b))

def assert_call_equal(call, expected_call, msg='', use_float_equal=False):

    msg = msg + ': ' if msg is not None else ''

    # function name
    assert call[0] == expected_call[0], '{} function name: {} vs {}'.format(msg, call[0], expected_call[0])

    function_name = call[0]

    # positional args
    assert_equal(call[1], expected_call[1], "{}func:'{}' {}".format(msg, function_name, 'positional args'), use_float_equal)

    # kword args
    assert_equal(call[2], expected_call[2], "{}func:'{}' {}".format(msg, function_name,  'kword args'), use_float_equal)


def float_equal(var1, var, name='', verbose=0):

    if var1 is None:
        assert  var is None
    else:
        assert  var is not None
    assert var1.size == var.size, 'shape not match {} vs {}'.format(var1.size, var.size)

    var1 = var1.flatten()
    var = var.flatten()
    diff = var1 - var
    diff_idx = np.where(diff != 0.0)
    if isinstance(diff_idx, tuple) and len(diff_idx[0]) > 0:
        NG_diff = abs(diff[diff_idx[0]])
        NG_diff = NG_diff[NG_diff >= 0.00001]
        if len(NG_diff) == 0:
            if verbose > 0:
                print('{} NGs are exists but maybe it known floating point operation error'.format(name))
            return True
        if verbose > 0:
            for i in range(len(var1)):
                if i in diff_idx[0]:
                    print(i, '{} {}'.format(name, 'NG'), var1[i], var[i])
                else:
                    print(i, '{} {}'.format(name, 'OK'), var1[i], var[i])
        return False
    else:
        if verbose > 0:
            print('{} OK'.format(name))
        return True
