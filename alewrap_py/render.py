
class Render(object):

    def __call__(self, frame):
        """
            params:
                frame : raw RGB frame
            return:
                rendered image
        """
        raise NotImplemented

    def reset(self, episode_id):
        self.episode_id = episode_id

    def step(self, frame, action, reward, term, info, episode_score, total_steps):
        """
            params:
                frame         : raw RGB frame wrt action
                action        : action index that agent selected wrt previous frame
                reward        : reward wrt action
                info          : information from environment with lives
                episode_score : current episode score
                total_steps   : total steps at recording
            return:
                void
        """
        raise NotImplemented

    def write_summmary(self, f):
        """
            params:
                f : file object for writable text file
            return:
                void
        """
        raise NotImplemented
