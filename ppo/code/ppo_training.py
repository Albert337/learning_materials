import gym
import os,datetime
import torch
import numpy as np
from ppo_agent import PPOAgent

Scene="Pendulum-v1"
env=gym.make(Scene)


NUM_EPOCHS=3000
NUM_STEPS=200
STATE_DIM=env.observation_space.shape[0]
ACTION_DIM=env.action_space.shape[0]
BATCH_SIZE=64
UPDATE_INTERVAL=BATCH_SIZE*2
REWARD_BUFFER=np.empty(shape=NUM_EPOCHS)

best_reward=-20000
current_path=os.path.dirname(os.path.realpath(__file__))
model = current_path+"/models/"
timestamp=datetime.datetime.now().strftime("%Y%m%d%H%M%S")


agent=PPOAgent(STATE_DIM,ACTION_DIM,BATCH_SIZE)  ##TODO

for epoch_i in range(NUM_EPOCHS):
    state,_=env.reset()
    done=False
    episode_reward=0


    for step_i in range(NUM_STEPS):
        action,value=agent.get_action(state)  #TODO
        observation,reward,done,truncated,info=env.step(action)
        episode_reward+=reward
        done=True if (step_i+1)==NUM_STEPS else False
        agent.replay_buffer.add_memo(state, action, value, done, reward)

        if done or (step_i+1)%UPDATE_INTERVAL==0:
            agent.update()  #TODO

    if episode_reward>=-100 and episode_reward>best_reward:
        best_reward=episode_reward
        agent.save_policy()  #TODO
        agent.save(agent.actor.state_dict(),model+f"ppo_actor+_{timestamp}.pth")
        print(f"Best reward: {best_reward}")

    REWARD_BUFFER[epoch_i]=episode_reward
    print(f"Episode: {epoch_i}, Reward: {round(episode_reward,2)}")

env.close()