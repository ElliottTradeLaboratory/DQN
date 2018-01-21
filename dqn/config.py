import os
import re
import sys
import datetime
from argparse import ArgumentParser

sys.path.append('../')
from utils import Namespace
    
def _parse_arguments():

    parser = ArgumentParser()

    parser.add_argument('--env',
                        type=str,
                        help="name of environment to use")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='random seed using random generator initializing.')
    parser.add_argument('--random_type',
                        type=str,
                        default='numpy',
                        choices=['numpy', 'pytorch', 'torch'],
                        help="specifies random generator type.")
    parser.add_argument('--lr',
                        type=float,
                        default=0.00025,
                        help='learning rate.')

    # agent options.
    parser.add_argument('--use_tie_break',
                        type=int,
                        choices=[1, 0],
                        default=1,
                        help="Specifies action selection behavior from q_value.")
    parser.add_argument('--use_regreedy',
                        type=int,
                        choices=[1, 0],
                        default=0,
                        help="Specifies action selection behavior from q_value.")

    # model architecture options.
    parser.add_argument('--backend',
                        choices=['tensorflow', 'mxnet', 'cntk', 'pytorch', 'pytorch_legacy', 'theano'],
                        default='pytorch_legacy',
                        type=str,
                        help='Specifies framework')
    parser.add_argument('--initializer',
                        type=str,
                        default='torch_nn_default',
                        help="Specifies parameter initializer.")
    parser.add_argument('--loss_function',
                        choices=['DQN3.0', 'huber', 'BFT1', 'BFT2'],
                        default='DQN3.0',
                        type=str,
                        help="Specifies loss function and gradients method.")
    parser.add_argument('--optimizer',
                        default='DQN3.0',
                        type=str,
                        choices=['DQN3.0', 'RMSpropCentered'],
                        help="Specifies optimizer.")
    parser.add_argument('--normalized_dqn',
                        action='store_true',
                        default=False,
                        help="Which use normalized_dqn or not.")
    parser.add_argument('--use_therano_grad',
                        action='store_true',
                        default=False,
                        help="Which use T.grad or not in backend='theano'")
    parser.add_argument('--relu',
                        action='store_true',
                        default=False,
                        help='Use the framework relu layer. As default, Use DQN3.0 Rectifier layer')
    parser.add_argument('--rectifier_div',
                        choices=[1, 0],
                        default=1,
                        type=int,
                        help='During compute output of Recrifier, 1: use devide 2.0, 0: use multiply 0.5')

    # preprocess options.
    parser.add_argument('--maximization',
                        choices=['env', 'agent', 'non'],
                        default='env',
                        help="Where to maximize.")
    parser.add_argument('--preproc',
                        type=str,
                        default='cv2',
                        help="Specifies use image processor in ['tensorlfow', 'PIL', 'cv2', 'image', 'scikit'] "
                        "or mix these, for example 'PIL:cv2' that means max(PIL output, cv2 output)." )
    parser.add_argument('--inter',
                        type=str,
                        default='AREA',
                        help="Specifies the interpolation at resizing depending on --preroc specified.")
    parser.add_argument('--screen_normalize',
                        type=str,
                        choices=['env', 'trans', 'none'],
                        default='trans',
                        help="Where does normalize the screen image. env: by env, trans: by transition table, non:doesn't normalize")

    # experiense sampling method option
    parser.add_argument('--simple_sampling',
                        action='store_true',
                        default=False,
                        help="For experience sampling, only use index sampling. By default, it excludes the terminal state in addition to that.")

    # environment option
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--step_train_mode',
                        type=int,
                        default=1,
                        choices=[1,0],
                        help="Which use the training mode same as DQN3.0 at step() during training or not. 1:use(default), 0:not use")
    parser.add_argument('--actrep',
                        default=4,
                        help="How many frames skip at every step().")
    parser.add_argument('--random_starts',
                        default=30,
                        help="How many frames skip when new game start.")
    parser.add_argument('--write_frame_to_png',
                        action='store_true',
                        default=False,
                        help="Save each screen image to a png. By default, save all screens in an episode into an AVI as moving image file.")



    parser.add_argument('--test',
                        action='store_true',
                        default=False,
                        help="run test mode.")
    parser.add_argument('--test_ep',
                        type=float,
                        default=0.05,
                        help="e-greedy epsilon in tes   t")
    parser.add_argument('--test_recording',
                        action='store_true',
                        default=False,
                        help="recording video in test mode.")
    parser.add_argument('--test_recording_q_value',
                        action='store_true',
                        default=False,
                        help="recording q_value in test mode.")
    parser.add_argument('--test_recording_q_value_show',
                        type=int,
                        default=0,
                        help="show recording q_value in test mode.")
    parser.add_argument('--test_episodes',
                        type=int,
                        default=30)
    parser.add_argument('--render',
                        action='store_true',
                        default=False,
                        help="Show the game screen on desktop.")
    parser.add_argument('--video_freq',
                        type=int,
                        default=100,
                        help="Video output frequency as default.")
    parser.add_argument('--logdir',
                        default='/home/deeplearning/work/tensorboard/logs',
                        help='log dir')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='use gpu #.')
    parser.add_argument('--not_use_egreedy',
                        action='store_true',
                        default=False)
    parser.add_argument('--file_name', 
                        type=str,
                        help='filename used for saving network and training history.')
    parser.add_argument('--run_memo', 
                        type=str,
                        help="If you specifies this, specified string is added to log dir name. it useful for identfy a type of model.")


    # Debug option
    parser.add_argument('--save_transitions_freq',
                        type=int,
                        default=0,
                        help='Save the scores in transitions. As default, does not save')
    parser.add_argument('--verbose',
                        type=int,
                        default=2,
                        help='the higher the level, the more information is printed to screen.')
    parser.add_argument('--debug',
                        action='store_true',
                        default=False)
    parser.add_argument('--training_summary',
                        type=str,
                        choices=['non', 'as_prot_freq', 'all'],
                        default='non')
    parser.add_argument('--steps',
                        type=int,
                        default=50000000)
    parser.add_argument('--log_device_placement',
                        action='store_true',
                        default=False,
                        help='Enable log_device_placement mode for tensorflow. other backends are not work')
    parser.add_argument('--tf_debug',
                        action='store_true',
                        default=False,
                        help='Enable tfdbg mode for tensorflow. other backends are not work')

    return parser.parse_known_args()

def _debug(opt):
    print('!!!!!!!!!!!!DEBUG MODE!!!!!!!!!!!!')
    opt.learn_start = 2000
    opt.eval_freq =  2000
    opt.eval_steps = 1000
    opt.save_freq =  1000

    opt.logdir = '/tmp'

    print("--logdir will reset to '/tmp' in debug mode.")
    
    return opt

def _dqn30_setting(opt):


    if isinstance(opt.actrep, str):
        assert re.match(r"^[\[\(][0-9],[0-9][)\]]$", opt.actrep), '--actrep is expected number or tuple eg.(2,5), but got {}.'.format(opt.actrep)
        opt.actrep = tuple([int(v) for v in re.split(r"[\(\)\,]", opt.actrep)[1:3]])

    # learning rate params (fixed)
    opt.lr = 0.00025
    opt.lr_start = opt.lr
    opt.lr_end   = opt.lr

    # input shape defines
    opt.input_height = 84
    opt.input_width = 84
    opt.state_dim = (opt.input_height * opt.input_width, )
    opt.ncols = 1
    opt.hist_len = 4
    opt.input_dims     = (opt.hist_len * opt.ncols, opt.input_width, opt.input_height)

    # transition table params
    opt.replay_memory = 1000000
    opt.bufferSize = 512
    opt.histType       = "linear"  # history type to use
    opt.histSpacing    = 1
    opt.nonTermProb    = 1
    
    # ε-greedy params
    opt.ep_end = 0.1
    opt.ep_start   = 1
    opt.ep_endt = opt.replay_memory

    opt.ep_restart = (opt.ep_start-opt.ep_end) / 2.0
    opt.ep_endt_restarted = int(opt.replay_memory / 2.0)
    opt.regreedy_threshold = 10
    opt.regreedy_rate = 0.9
    opt.regreedy_ema_momemtum = 0.9

    # RMSprop params
    opt.grad_momentum = 0.95
    opt.sqared_grad_momentum = 0.95
    opt.mini_squared_gradient = 0.01
    opt.momentum = 0
    opt.wc = 0
    
    # varidation params
    opt.valid_size = 500

    # training params
    opt.minibatch_size = 32
    opt.clip_delta = 1
    opt.clip_reward = 1
    opt.rescale_r = True
    opt.discount = 0.99
    opt.n_replay = 1
    opt.update_freq = 4

    # frequencies
    opt.target_q = 10000
    opt.learn_start = 50000
    opt.eval_freq =  250000
    opt.eval_steps = 125000
    opt.save_freq =  250000
    opt.prog_freq =   1000

    opt.num_threads = 1
    
    return opt


def get_opt():

    args, _ = _parse_arguments()
    
    opt = Namespace(vars(args))

    assert opt.env is not None
    assert opt.backend is not None

    #-------------------------------------------
    # default core setting is same as DQN3.0
    opt = _dqn30_setting(opt)

    #-------------------------------------------
    # debug mode setting
    if opt.debug:
        opt = _debug(opt)

    #-------------------------------------------
    # log dir set up.
    from datetime import datetime

    datetime_str = datetime.now().strftime('%Y%m%d-%H%M%S')

    opt.run_name = "{}-{}{}-{}".format(
                        opt.env,
                        opt.backend,
                        '-{}'.format(opt.run_memo) if opt.run_memo else '',
                        datetime_str)
    opt.log_dir = "{}/{}/{}".format(opt.logdir, opt.env, opt.run_name)
    opt.monitor_dir = "{}/monitor".format(opt.log_dir)
    os.makedirs(opt.log_dir)

    with open(opt.log_dir + '/config.txt', 'wt') as f:
        opt.summary(f)

    #-------------------------------------------
    # simple logger set up.
    from utils import Unbuffered
    sys.stdout = Unbuffered(
                    sys.stdout,
                    open('{}/stdout_{}.log'.format(opt.log_dir, datetime_str), "wt")
                 )

    sys.stderr = Unbuffered(
                    sys.stderr,
                    open('{}/stderr_{}.log'.format(opt.log_dir, datetime_str), "wt")
                 )


    #-------------------------------------------
    # framework setup before loading
    from common import get_extensions
    Ex = get_extensions(opt)
    Ex.setup_before_package_loading(opt)

    return opt
