from collections import deque
import numpy as np
from utils import get_random

class TransitionTable:
    
    def __init__(self, args, shapes, non_term_prob=1, non_event_prob=1):

        self.args  = args
        self.shapes = shapes
        self.bufferSize = args.bufferSize
        self.non_term_prob = non_term_prob
        self.non_event_prob = non_event_prob
        self.maxSize = args.replay_memory
        self.hist_len = args.hist_len
        self.numEntries = 0
        self.buf_ind = -1
        self.insertIndex = 0

        if not args.test:
            self.s = np.zeros((self.maxSize,) + self.shapes['s'], dtype=np.uint8)
            self.a = np.zeros((self.maxSize,), dtype=np.uint8)
            self.r = np.zeros((self.maxSize,), dtype=np.int8 if args.clip_reward else np.int16)
            self.t = np.zeros((self.maxSize,), dtype=np.uint8)

        self.recent_s = np.zeros((self.hist_len,) + self.shapes['s'], dtype=np.float32)
        self.recent_t = np.zeros((self.hist_len,), dtype=np.uint8)

        self.buf_s = np.zeros((self.bufferSize,) + (self.hist_len,) + self.shapes['s'], dtype=np.float32)
        self.buf_a = np.zeros((self.bufferSize,), dtype=np.uint8)
        self.buf_r = np.zeros((self.bufferSize,), dtype=np.int8 if args.clip_reward else np.int16)
        self.buf_s2 = np.zeros((self.bufferSize,) + (self.hist_len,) + self.shapes['s'], dtype=np.float32)
        self.buf_term = np.zeros((self.bufferSize,), dtype=np.uint8)

        self.last_s = None

        self.sample_index = []

        self.random = get_random()
        
        def normalize(s):
            return s.astype(np.float32) / 255.0
        def nop_normalize(s):
            return s.astype(np.float32)
        def denormalize(s):
            return np.uint8(s * 255.0)
        def nop_denormalize(s):
            return s.astype(np.uint8)

        normalize_method = {'env'  :(denormalize, normalize),
                            'trans':(nop_denormalize, normalize),
                            'none' :(nop_denormalize, nop_normalize)}
        self.denormalize, self.normalize = normalize_method[self.args.get('screen_normalize', 'env')]

    def _fill_buffer(self):
        assert self.numEntries >= self.bufferSize
        #print('%%%%%%%%%%_fill_buffer()%%%%%%%%%%%')

        self.buf_ind = 0

        for buf_ind in range(self.bufferSize):
            s, a, r, s2, t = self._sample_one()

            self.buf_s[buf_ind] = np.copy(s).astype(np.float32)
            self.buf_a[buf_ind] = a
            self.buf_r[buf_ind] = r
            self.buf_s2[buf_ind] = np.copy(s2).astype(np.float32)
            self.buf_term[buf_ind] = t

        self.buf_s = self.normalize(self.buf_s)
        self.buf_s2 = self.normalize(self.buf_s2)

    def _sample_one(self):
        assert self.numEntries >= self.hist_len
        index = 0
        valid = False
        while not valid:

            # Reason of subtracting 1 from random value is for validate exactly same to DQN3.0.
            index = self.random.random(2, self.numEntries - self.hist_len)-1

            if self.t[index + self.hist_len-1] == 0:
                valid = True
                
            """
            if self.non_term_prob < 1 and t == 0 and self.random.uniform() > self.non_term_prob:
                valid = False
                
            if self.non_event_prob < 1  and r == 0 and self.random.uniform() > self.non_event_prob:
                valid = False

            """
        return self._get(index)


    def sample(self, batch_size):
        assert batch_size < self.bufferSize

        #print('%%%%%%%%%%sample() batch_size={}%%%%%%%%%%%'.format(batch_size))
        if self.buf_ind == -1 or self.buf_ind + batch_size - 1> self.bufferSize:
            self._fill_buffer()

        index = self.buf_ind

        self.buf_ind = self.buf_ind+batch_size
        last_index = index + batch_size
        #print('index=',index,'last_index=',last_index)

        buf_s, buf_s2, buf_a, buf_r, buf_term = self.buf_s, self.buf_s2,\
                                                self.buf_a, self.buf_r, self.buf_term

        #print(buf_a[index:last_index])
        #print(self.buf_a[index:last_index])

        return buf_s[index:last_index],\
               buf_a[index:last_index],\
               buf_r[index:last_index],\
               buf_s2[index:last_index],\
               buf_term[index:last_index],\

    def _concatFrames(self, index, use_recent):
        
        if use_recent:
            s, t = self.recent_s, self.recent_t
        else:
            s, t = self.s, self.t

        zero_out = False
        j = 0

        episode_start = self.hist_len - 1

        for i in range(self.hist_len - 2, -1, -1):
            if zero_out or t[index + i] == 1:
                zero_out = True
            else:
                episode_start = i

        fullstate = np.zeros((self.hist_len,)+ self.shapes['s'], dtype=np.uint8)

        for j in range(episode_start, self.hist_len):
            fullstate[j,...] = s[index + j]

        return fullstate
        
    def get_recent(self):
        state = self._concatFrames(0, True)
        return self.normalize(state.astype(np.float32))

    def _get(self, index):
        s = self._concatFrames(index, False)
        s2 = self._concatFrames(index + 1, False)
        ar_index = index + self.hist_len - 1

        return s, self.a[ar_index], self.r[ar_index], s2, self.t[ar_index+1]

    def add(self, s, a, r, t):
        s = self.denormalize(s)

        self.s[self.insertIndex,...] = s
        self.a[self.insertIndex] = a
        self.r[self.insertIndex] = r
        self.t[self.insertIndex] = t * 1

        if self.numEntries < self.maxSize:
            self.numEntries += 1

        if self.insertIndex + 1 >= self.maxSize:
            self.insertIndex  = 0
        else:
            self.insertIndex += 1


    def add_recent_state(self, s, t):

        self.recent_s = np.roll(self.recent_s, -1, axis=0)
        self.recent_s[-1,...] = self.denormalize(s)
        self.recent_t = np.roll(self.recent_t, -1)
        self.recent_t[-1] = t * 1

