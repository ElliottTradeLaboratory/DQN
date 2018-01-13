import os, json, logging, numpy as np, six
from collections import Counter
import skvideo.io
import cv2
from .environments import BaseEnv

FILE_PREFIX = 'RecorderEnv'
MANIFEST_PREFIX = FILE_PREFIX + '.manifest'

        
class RecorderEnv(BaseEnv):
    def __init__(self, env, args):
        super(RecorderEnv, self).__init__(args)
        self.args = args
        self.verbose = args.get('verbose', 2)
        if self.verbose >= 10:
            print('RecorderEnv.__init__')

        self.write_frame_to_png = args.get('write_frame_to_png', False)
        self.video_freq = args.get('video_freq', 100)
        self.env = env

        modes = getattr(env, 'metadata', {'render.modes':[]}).get('render.modes', [])
        ansi_mode = False
        if 'rgb_array' not in modes:
            if 'ansi' in modes:
                self.ansi_mode = True
            else:
                raise ValueError('must set "render.modes" in env.metadata')
        self.render_mode = 'ansi' if ansi_mode else 'rgb_array'

        self.actions = self.getActions()
        self.n_actions = len(self.actions)

        self._frame_shape = env.frame_shape

        self._init_state(False)

    def _init_state(self, recording_mode):

        self.recording_mode = recording_mode

        self.max_episode_score = self.args.initial_max_episode_score \
                                    if hasattr(self.args, 'initial_max_episode_score') and \
                                       getattr(self.args, 'initial_max_episode_score') is not None \
                                    else 0
        self.action = 0
        self.episode_score = 0
        self.episode_scores = np.zeros((1000,), dtype=np.int16)
        self.episode_id = 0
        self.total_steps = 0

        self._clear_video()


    def getState(self):

        self._episode_end_process()
        observation, r, t, info = self.env.getState()
        self._after_reset(observation, r, t, info)

        return observation, r, t, info 

    def step(self, action, training):

        self._before_step(action, training)

        observation, reward, done, info = self.env.step(action, training)

        done = self._after_step(observation, reward, done, info)

        return observation, reward, done, info

    def newGame(self):
            
        self._episode_end_process()
        observation, reward, done, info = self.env.newGame()
        done = self._after_reset(observation, reward, done, info)

        return observation, reward, done, info 

    def nextRandomGame(self):
        
        self._episode_end_process()
        observation, reward, done, info = self.env.nextRandomGame()
        done = self._after_reset(observation, reward, done, info)

        return observation, reward, done, info

    def start_recording(self, numSteps, render=None):

        self.recording_mode =True
        self._init_state(True)

        self.render = render

        self.out_dir = '{}/step_{:010d}'.format(self.args.get('monitor_dir'), numSteps)
        self._start_video(self.out_dir)

    def getActions(self):
        return self.env.getActions()


    def stop_recording(self):
        self._clear_video()
        self._write_summary(self.out_dir)
        self.episode_id = 0
        self.recording_mode =False
        self.total_steps = 0
        self.episode_score = 0
        self.max_episode_score = 0

    def _episode_end_process(self):

        if self.recording_mode:
            if self.episode_id > 0:
                # episode_id equal zero is first call of newGame() at test or train

                flush = False

                if self.episode_score > self.max_episode_score:
                    self.max_episode_score = self.episode_score
                    flush = True

                elif self.episode_id % self.video_freq == 0:
                    flush = True

                if flush:
                    if self.write_frame_to_png:
                        self._flush_png(self.episode_id, self.episode_score)
                    else:
                        self._flush_video(self.episode_id, self.episode_score)

                self._clear_video()

                idx = self.episode_id - 1
                if idx >= len(self.episode_scores):
                    self.episode_scores = np.concatnate((self.episode_scores, np.zeros((1000,0), dtype=np.int16)), axis=0)

                self.episode_scores[idx] = self.episode_score

            self.episode_id += 1
        self.episode_score = 0

    def _after_reset(self, observation, reward, done, info):
        if self.render:
            self.render.reset(self.episode_id)

        return self._after_process(observation, reward, done, info)

    def _before_step(self, action, training):

        self.action = self.actions.index(action)

    def _after_step(self, observation, reward, done, info):

        self.episode_score += reward

        done = self._after_process(observation, reward, done, info)

        if self.recording_mode:
            self.total_steps += 1

        return done
        
    def _after_process(self, observation, reward, done, info):
        if self.render:
            self.render.step(observation,
                             self.action,
                             reward,
                             done,
                             info,
                             self.episode_score,
                             self.total_steps)

        if self.recording_mode:
            self._capture_frame()
        return done

    def _write_summary(self, directory):

        summary_filepath = os.path.join(directory, 'summary.txt')

        with open(summary_filepath, "w") as f:
            f.write('episode:score\n')
            episode_scores = self.episode_scores[:self.episode_id-1]
            for i, score in enumerate(episode_scores):
                f.write('{}:{}\n'.format(i+1, score))
            f.write('---------------------------\n')
            f.write('sumamry\n')
            f.write('---------------------------\n')
            f.write('total      :{}\n'.format(sum(episode_scores)))
            f.write('average    :{}\n'.format(np.mean(episode_scores)))
            f.write('max        :{}\n'.format(max(episode_scores)))
            f.write('min        :{}\n'.format(min(episode_scores)))
            f.write('median     :{}\n'.format(np.median(episode_scores)))
            f.write('variance   :{}\n'.format(np.var(episode_scores)))
            f.write('stdev      :{}\n'.format(np.std(episode_scores, ddof=1)))
            f.write('most common:\n')
            f.write('---------------------------\n')

            c = Counter(episode_scores)
            for most_common in sorted(c.most_common(),
                                       key=lambda x:x[0],
                                       reverse=True):
                f.write('{}:{}\n'.format(*most_common))

            if self.render:
                self.render.write_summmary(f)

    def _start_video(self, directory):

        self.directory = directory

        env_id = self.env.game_name

        if not os.path.exists(directory):
            if six.PY3:
                os.makedirs(directory, exist_ok=True)
            else:
                os.makedirs(directory)

    def _capture_frame(self):
        frame = self.env.render(mode=self.render_mode)

        if self.render:
            frame = self.render(frame)

        if self._frame_shape != frame.shape:
            self._frame_shape = frame.shape
            self.frames = None
            self._clear_video()

        if self.frame_index >= self.frames.shape[0]:
            self.frames = np.concatenate((self.frames, np.zeros((1000,)+frame.shape, dtype=np.uint8)), axis=0)
 
        self.frames[self.frame_index,...] = frame
        self.frame_index += 1


    def _flush_video(self, episode_id, episode_score):

        if not hasattr(self, 'frames') or len(self.frames) == 0:
            return

        filename = os.path.join(self.directory, 'episode={:010d}_score={:010d}.avi'.format(episode_id, int(episode_score)))

        skvideo.io.vwrite(filename, self.frames[:self.frame_index])

        assert os.path.exists(filename)

        print('Write video {}'.format(filename))

    def _flush_png(self, episode_id, episode_score):

        if not hasattr(self, 'frames') or len(self.frames) == 0:
            return

        for i, frame in  enumerate(self.frames[:self.frame_index]):
            filename = os.path.join(self.directory, 'episode={:010d}_score={:010d}_{:050}.png'.format(episode_id, int(episode_score), i+1))
            cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            assert os.path.exists(filename)

        print('Write pngs')

    def _clear_video(self):
        if getattr(self, 'frames', None) is None:
            self.frames = np.empty((5000,) + self._frame_shape, dtype=np.uint8)
        # NOTE: For performance, frames will not clear.
        # it will be clear only frame_index.
        self.frame_index = 0

    def __del__(self):
        pass

    def get_total_steps(self):
        return self.total_steps

    def get_episode_scores(self):
        return self.episode_scores

    def get_num_episode(self):
        return len(self.episode_scores)

    def render(self, mode='human', close=False):
        self.env.render(mode, close)

    @property
    def frame_shape(self):
        return self._frame_shape
