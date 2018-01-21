import sys
import unittest
from unittest.mock import *

import numpy as np
import cv2

from torch.utils.serialization import load_lua

if sys.path.count('../') == 0:
    sys.path.append('../')
if sys.path.count('../../') == 0:
    sys.path.append('../../')
    
from testutils import *
from utils import Namespace

class TestScale(unittest.TestCase):

    def test_01_image_01(self):
    
        from common import get_preprocess
        
        preproc = get_preprocess('image')(84, 84, Namespace(inter='LINEAR'))
        
        img = cv2.imread('./test_img/screen.png')
        #preproc.show(img)
        
        s = preproc.forward(img)
        
        dqn_s = load_lua('./dqn3.0_dump/scale.dat').numpy()
        dqn_s = dqn_s.reshape(84, 84)
        #dqn_s = np.transpose(dqn_s, (1,2,0)).reshape(84,84)

        #preproc.show(s)
        #preproc.show(dqn_s)
        #assert float_equal(s, dqn_s, verbose=10)

    def test_02_cv2_01(self):
    
        from common import get_preprocess
        
        preproc = get_preprocess('cv2')(84, 84, Namespace(inter='LINEAR'))
        
        img = cv2.imread('./test_img/screen.png')
        
        s = preproc.forward(img.astype(np.float32) / 255.0)
        
        y = cv2.resize(cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2GRAY), (84, 84), cv2.INTER_LINEAR)

        assert float_equal(s, y, verbose=10)

    def test_03(self):
        preprocs = ['cv2', 'tensorflow', 'PIL', 'image', 'scikit']
        inters = {'cv2':['NEAREST','LINEAR','AREA','CUBIC','LANCZOS4'],
                 'tensorflow': ['NEAREST', 'LINEAR','AREA','CUBIC'],
                 'PIL':['NEAREST','LINEAR','BOX','CUBIC','LANCZOS4','HAMMING'],
                 'scikit':['NEAREST','LINEAR','QUADRATIC','CUBIC','QUARTIC','QUINTIC'],
                 'image':['bilinear']}
        
        from common import get_preprocess
        from utils import get_random
        from alewrap_py import get_env
        import cv2
        
        get_random('pytorch', 1)
        env = get_env(dict(env='space_invaders',actrep=4, screen_normalize='env', maximization='env'))
        s, r, t, info =env.getState()
        
        print(s.max(), s.min())
        
        for preproc in preprocs:
            inter = inters[preproc]
            
            for intr in inter:
                with self.subTest('{}-{}'.format(preproc, intr)):
                    scale = get_preprocess(preproc)(84,84,Namespace(inter=intr, gpu=-1))
                    x = scale.forward(s)
                    cv2.imwrite('tmp/{}_{}.png'.format(preproc, intr), np.uint8(x * 255.0))

    def test_04(self):
        preprocs = ['PIL:cv2']

        from common import get_preprocess
        from utils import get_random
        from alewrap_py import get_env
        import cv2
        
        get_random('pytorch', 1)
        env = get_env(dict(env='space_invaders',actrep=4))
        s, r, t, info =env.getState()
        
        for preproc in preprocs:
            
            for intr in ['LINEAR']:
                with self.subTest('{}-{}'.format(preproc, intr)):
                    scale = get_preprocess(preproc)(84,84,Namespace(inter=intr,
                                                                    gpu=-1,
                                                                    preproc=preproc,
                                                                    gain_level=0.2,
                                                                    gain=1.5))
                    x = scale.forward(s)
                    print(x.shape)
                    cv2.imwrite('tmp/{}_{}.png'.format(preproc.replace(':','-'), intr), np.uint8(x * 255.0))

    def test_05(self):
        from common import get_preprocess
        from utils import get_random
        from alewrap_py import get_env
        import cv2
        
        get_random('pytorch', 1)
        env = get_env(dict(env='breakout',actrep=4, screen_normalize='env', maximization='env'))
        s, r, t, info =env.getState()
        
        args =Namespace(inter='LINEAR',
                        gpu=0,
                        preproc='image',
                        screen_normalize='env')
        args.preproc = 'image'
        scale_image = get_preprocess('image')(84,84, args)
        image_x = scale_image.forward(s)

        preprocs = ['cv2', 'tensorflow', 'PIL', 'scikit']
        inters = {'cv2':['NEAREST','LINEAR','AREA','CUBIC','LANCZOS4'],
                 'tensorflow': ['NEAREST', 'LINEAR','AREA','CUBIC'],
                 'PIL':['NEAREST','LINEAR','BOX','CUBIC','LANCZOS4','HAMMING'],
                 'scikit':['NEAREST','LINEAR','QUADRATIC','CUBIC','QUARTIC','QUINTIC'],
                 'image':['bilinear']}

        print(np.mean(s), s.dtype)

        diffs = []
        for preproc in preprocs:
            inter = inters[preproc]
            
            for intr in inter:
                with self.subTest('{}-{}'.format(preproc, intr)):
                    args.preproc = preproc
                    args.inter = intr
                    scale_image = get_preprocess(preproc)(84,84, args)
                    x = scale_image.forward(s)
                    print('image vs {}-{}:mean:{:.10f} var:{:.10f} std:{:.10f},MSE:{:.10f}'.format(preproc, intr,
                                                              np.mean(image_x-x),
                                                              np.var(image_x-x),
                                                              np.std(image_x-x, ddof=1),
                                                              np.mean((image_x-x)**2)
                                                              ))
                    diffs.append(['{}-{}'.format(preproc, intr), np.mean((image_x-x)**2)])

        diffs = np.array(diffs)
        diffnum = np.absolute([float(s) for s in diffs[...,1]])
        index = np.argmin(diffnum)
        print('minimum diff {} {}'.format(*diffs[index]))
    
    """
    def test_06(self):
        import os
        import ale_python_interface
        from common import get_preprocess
        from alewrap_py import get_env
        from utils import get_random
        get_random('pytorch', 1)

        ale = ale_python_interface.ALEInterface()
        ale.loadROM(os.path.join("/home/deeplearning/git/atari-py/atari_py/atari_roms", "breakout.bin"))
        screen_RGB = ale.getScreenRGB()
        screen_gray = ale.getScreenGrayscale()

        cv2.imwrite('tmp/ALE_RGB.png', cv2.cvtColor(screen_RGB, cv2.COLOR_BGR2RGB))
        cv2.imwrite('tmp/ALE_gray.png', screen_gray)

        env = get_env(dict(env='breakout',actrep=4))
        s, r, t, info =env.getState()
        s = np.uint8(s*255.0)
        assert_equal(s, screen_RGB)

        args =Namespace(inter='LINEAR',
                        gpu=0,
                        preproc='image')
        scale_image = get_preprocess('image')(84,84, args)
        cv2.imwrite('tmp/atari_py_RGB.png', cv2.cvtColor(s, cv2.COLOR_BGR2RGB))
        cv2.imwrite('tmp/atari_py_gray.png', scale_image.rgb2y(s))
        
        assert_equal(scale_image.rgb2y(screen_RGB), scale_image.rgb2y(s))
        grayscaled_rgb = np.uint8(scale_image.rgb2y(screen_RGB))
        assert_equal(screen_gray, grayscaled_rgb, verbose=1, use_float_equal=True)
        """

class TestScaleTensorflow(unittest.TestCase):
    def setUp(self):
        from config import get_opt
        from time import sleep
        from utils import get_random
        sleep(1)
        sys.argv += ['--backend', 'pytorch_legacy', '--env', 'breakout', '--debug', '--inter', 'AREA']
        self.opt = get_opt()
        get_random('pytorch', 1)

    def test_01_screen_normalize_env(self):
        from alewrap_py import get_env
        from common import get_preprocess
        
        self.opt.screen_normalize = 'env'
        
        env = get_env(self.opt)
        preproc = get_preprocess('tensorflow')(84, 84, self.opt)
        
        s, a, r, info = env.step(0, True)
        
        preproc.show(np.uint8(s*255.0))
        print(np.mean(s))
        assert np.all(s <= 1.0)
        assert np.all(s >= 0.0)
        
        new_s = preproc.forward(s)
        preproc.show(np.uint8(new_s*255.0))
        print(np.mean(new_s))

        assert np.all(new_s <= 1.0)
        assert np.all(new_s >= 0.0)

    def test_02_screen_normalize_tran(self):
        from alewrap_py import get_env
        from common import get_preprocess
        
        self.opt.screen_normalize = 'trans'
        
        env = get_env(self.opt)
        preproc = get_preprocess('tensorflow')(84, 84, self.opt)
        
        s, a, r, info = env.step(0, True)

        print(np.mean(s))
        preproc.show(np.uint8(s))
        assert np.all(s <= 255.0)
        assert np.all(s >= 0.0)
        
        new_s = preproc.forward(s)
        preproc.show(np.uint8(new_s))
        print(np.mean(new_s))

        assert np.all(new_s <= 255.0)
        assert np.all(new_s >= 0.0)
        