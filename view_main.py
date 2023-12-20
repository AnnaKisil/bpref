import utils
from reward_model import RewardModel
from agent.sac import SACAgent
import hydra
import os
import gym
import pybullet
from gym.wrappers.record_video import RecordVideo
from gym.wrappers.monitoring.video_recorder import VideoRecorder



@hydra.main(config_path='config/train_PEBBLE.yaml', strict=True)
def main(cfg):
    model_dir = os.getcwd()
    vid_path = os.path.join(model_dir, 'video_test.mp4')
    env = utils.make_metaworld_env(cfg)
    env = RecordVideo(env, 'video_bpref')
    # env.render_mode = 'rgb_array'

    # vid_recorder = VideoRecorder(env, vid_path, enabled=True)

    step = 1000000
    obs_space = env.observation_space.shape[0]
    act_space = env.action_space.shape[0]

    model = RewardModel(obs_space, act_space)
    model.load(model_dir=model_dir, step=step)

    cfg.agent.params.obs_dim = env.observation_space.shape[0]
    cfg.agent.params.action_dim = env.action_space.shape[0]
    cfg.agent.params.action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())
    ]

    agent = hydra.utils.instantiate(cfg.agent)
    agent.load(model_dir=model_dir, step=step)

    # obs = env.reset()

    # for i in range(1000):
    #     env.render()
    #     action = agent.act(obs)
    #     vid_recorder.capture_frame()
    #     obs, reward, done, info = env.step(action)
    #     print(info)

    #     if done:
    #         print('done')

    #         obs = env.reset()
    #         agent.reset()
    #         vid_recorder.close()
    #         vid_recorder.enable = False
    modes = env.metadata
    print(modes)
    obs = env.reset()
    env.start_video_recorder()
    print('1')

    for i in range(10):
        action = agent.act(obs)

        obs, reward, done, info = env.step(action)

        env.render()
        
        # print(info)

        if done:
            print('done')

            obs = env.reset()
            agent.reset()
            env.close()




if __name__ == '__main__':
    main()
