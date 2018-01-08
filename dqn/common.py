_Ex = None
def get_extensions(opt):
    global _Ex
    if _Ex == None:
        if opt.backend == 'tensorflow':
            import tensorflow_extensions as Ex
        elif opt.backend == 'cntk':
            import cntk_extensions as Ex
        elif opt.backend in ['pytorch', 'pytorch_legacy']:
            import pytorch_extensions as Ex
        elif opt.backend == 'mxnet':
            import mxnet_extensions as Ex
        elif opt.backend == 'theano':
            import theano_extensions as Ex

        _Ex = Ex

    return _Ex

def create_networks(args):
    if args.backend in ['pytorch', 'pytorch_legacy']:
        from convnet_pytorch import create_networks
    elif args.backend in ['tensorflow', 'cntk', 'theano']:
        from convnet_keras   import create_networks
    elif args.backend == 'mxnet':
        from convnet_mxnet   import create_networks

    return create_networks(args)

def get_preprocess(type):
    from scale import get_preprocess as _get_preprocess
    return _get_preprocess(type)


