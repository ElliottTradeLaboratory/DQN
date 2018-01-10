import sys
import time
from datetime import datetime
from collections import deque
import numpy as np

sys.path.append('../')
from utils import get_random
from config import get_opt
from visualizer import CurrentStateVisualizer, EvaluationVisualizer

def get_device_name(opt):
    if opt.gpu >= 0:
        return '/gpu:{}'.format(opt.gpu)
    else:
        return None

def train_main(game_env, agent, game_actions, opt):

    learn_start = opt.learn_start
    start_time = time.time()
    time_history = deque(maxlen=10)
    reward_history = []
    step = 0
    num_epi = 0
    num_frame = 0
    time_history.append(0)
    cur_epi_count = 0
    cur_epi_reward = 0
    cur_total_epi_reward = 0
    cur_total_epi_count = 0
    total_epi_reward = 0
    max_epi_reward = 0
    total_max_reward = 0
    action_index = 0
    current_state_visualizer = CurrentStateVisualizer(opt)
    eval_visualizer = EvaluationVisualizer(opt)

    get_random().manualSeed(1)

    screen, reward, terminal, info = game_env.getState();

    start_datetime = datetime.now()
    
    last_screen = None

    print("Iteration ..", opt.steps, start_datetime.strftime('%Y-%m-%d %H:%M:%S'))
    while step < opt.steps:
        step += 1

        if opt.render:
            game_env.render()

        action_index = agent.perceive(screen, reward, terminal, testing=False, testing_ep=None)

        if not terminal:
            screen, reward, terminal, info = game_env.step(game_actions[action_index], training=opt.step_train_mode)

            cur_epi_reward += reward
            if 'frameskip' in info:
                num_frame += info['frameskip']
            else:
                num_frame += np.mean(opt.actrep) if isinstance(opt.actrep, tuple) else opt.actrep
        else:
            num_epi += 1
            cur_epi_count += 1
            cur_total_epi_reward += cur_epi_reward
            max_epi_reward = max(max_epi_reward, cur_epi_reward)
            cur_epi_reward = 0

            if opt.random_starts > 0 :
                screen, reward, terminal, info = game_env.nextRandomGame()
            else:
                screen, reward, terminal, info = game_env.newGame()


        if step % opt.prog_freq == 0 :
            assert step==agent.numSteps, 'trainer step: {0} & agent.numSteps: {1}'.format(step,agent.numSteps)
            print('------------------------------------------------')
            cur_avg_epi_reward = max(cur_total_epi_reward, cur_epi_reward) / max(1, cur_epi_count)
            cur_total_epi_count = cur_total_epi_count + cur_epi_count
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Steps: ", step, "epsilon:{:.2f}".format(agent.ep), "Episodes:", num_epi, 'total_epi_reward:', cur_total_epi_reward, 'max_epi_reward:', max_epi_reward, 'avg_epi_reward:{:.2f}'.format(cur_avg_epi_reward))
            current_state = dict(
                average_episode_scores = cur_avg_epi_reward,
                episode_count = cur_total_epi_count,
                epsilon = agent.ep
            )
            current_state_visualizer.addCurrentState(current_state)
            current_state_visualizer.flush(step)
            
            cur_epi_reward = 0
            cur_total_epi_reward = 0
            cur_epi_count = 0
            max_epi_reward = 0

        if step % opt.eval_freq == 0 and step > learn_start:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'step[{}] eval start'.format(step))

            total_reward = 0
            nrewards = 0
            nepisodes = 0
            episode_reward = 0
            eval_frames = 0
            score_list = []
            eval_time = time.time()

            game_env.start_recording(step)

            screen, reward, terminal, info = game_env.newGame()
            
            for estep in range(opt.eval_steps):

                action = agent.perceive(screen, reward, terminal, testing=True, testing_ep=0.05)

                # Play game in test mode (episodes don't end when losing a life)
                screen, reward, terminal, info = game_env.step(game_actions[action], training=False)

                if 'frameskip' in info:
                    eval_frames += info['frameskip']
                else:
                    eval_frames += opt.actrep
                
                if terminal:
                    print('episode:{} score:{}'.format(game_env.episode_id, game_env.episode_score))
                    screen, reward, terminal, info = game_env.nextRandomGame()
                    
                
                if estep % opt.prog_freq == 0 :
                    print('eval steps {0}/{1}'.format(estep, opt.eval_steps))
            
            game_env.stop_recording()

            total_reward = sum(game_env.episode_scores)
            nepisodes = len(game_env.episode_scores)
            avg_epi_reward = np.mean(game_env.episode_scores)
 
            agent.decide_regreedy(avg_epi_reward)
 
            eval_time = time.time() - eval_time
            start_time += eval_time
            agent.compute_validation_statistics()
            
            eval_values = dict(episode_score=np.array(game_env.episode_scores), episode_count=nepisodes)
            valid_values = dict(TDerror=agent.tderr_avg, V=agent.v_avg)
            eval_visualizer.addEvaluation(eval_values)
            eval_visualizer.addValidation(valid_values)
            eval_visualizer.flush(step)

            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'V {0} TD error {1} Qmax {2}'.format(agent.v_avg, agent.tderr_avg, agent.q_max))
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'episode_count:{0} total_epi_reward:{1:d} avg_epi_reward:{2:.2f}'.format(nepisodes, int(total_reward), total_reward / nepisodes))
            
            reward_history.append(total_reward)
            
            time_history.append(time.time() - start_time)
            
            last_idx = 0
            time_dif = 0

            training_rate =0
            
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Steps: {0} (frames: {1}), reward: {2:.2f}, epsiron: {3:.2f}, lr:{4:.5f}, training time: {5}s, training rate: {6}fps, testing time: {7}s testing rate: {8}fps, num. ep.: {9}, num. rewards: {10}'.format(
                        step,
                        num_frame,
                        avg_epi_reward,
                        0.05,
                        agent.lr,
                        int(time_dif),
                        int(training_rate),
                        eval_time,
                        int(eval_frames / eval_time),
                        nepisodes,
                        nrewards))

        if step % opt.save_freq == 0 or step == opt.steps:
            filepath = '{}/{}_{}_network_step{:010d}.dat'.format(opt.log_dir, opt.env, opt.backend, step)
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'step[{0}] save network to [{1}]'.format(step, filepath))
            agent.save_network(filepath)

            

        sys.stdout.flush()
        

def test_main(game_env, agent, game_actions, opt):

    total_reward = 0
    nrewards = 0
    nepisodes = 0
    episode_reward = 0
    episode_rewards = []

    eval_time = time.time()
    
    print('--------------------------------------------------------------')
    print('test start',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    screen, reward, terminal, info = game_env.getState()

    while nepisodes < opt.test_episodes:
    
        action = agent.perceive(screen, reward, terminal, True, 0.05)

        screen, reward, terminal, info = game_env.step(game_actions[action], False)

        if opt.render:
            game_env.render()

        episode_reward = episode_reward + reward
        
        if terminal:
            print('episode:{:03d} reward:{}'.format(nepisodes+1, episode_reward))
            episode_rewards.append(episode_reward)
            episode_reward = 0
            nepisodes = nepisodes + 1

            screen, reward, terminal, info = game_env.nextRandomGame()
            
    episode_rewards = np.array(episode_rewards)
    print('test end', datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total reward: {:.2f}'.format(episode_rewards.sum()),
            'average reward: {:.2f}'.format(episode_rewards.mean()),
            'stdv: {:.2f}'.format(episode_rewards.std(ddof=1)),
            )

def main():
    import subprocess
    import os

    opt = get_opt()

    from initenv import setup
    game_env, agent, game_actions, opt = setup(opt)

    if opt.test:
        test_main(game_env, agent, game_actions, opt)
    else:
        rootdir = os.getcwd().replace("/dqn","")
        subprocess.run([os.path.join(rootdir,"copy_source.sh"), rootdir, opt.log_dir])
        train_main(game_env, agent, game_actions, opt)

    del game_env


if __name__ == '__main__':
    
    main()
