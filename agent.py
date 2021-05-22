import gym
import torch.multiprocessing as mp

from model import ActorCritic, SharedAdam

EPISODES = 10000
T_MAX = 5


class Agent(mp.Process):

    def __init__(self, global_actor_critic, optimizer, input_dims, nb_actions, gamma, lr, name, global_ep_index,
                 env_id):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, nb_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = "w%02i" % name
        self.episode_index = global_ep_index
        self.env = gym.make(env_id)
        self.optimizer = optimizer

    def run(self):
        t_step = 1
        while self.episode_index.value < EPISODES:
            done = False
            observation = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action = self.local_actor_critic.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if (t_step % T_MAX) == 0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()
                t_step += 1
                observation = observation_
            with self.episode_index.get_lock():
                self.episode_index.value += 1
            print(self.name, 'episode ', self.episode_index.value, 'reward %.1f' % score)


if __name__ == '__main__':
    lr = 1e-4
    env_id = 'CartPole-v0'
    nb_actions = 2
    input_dims = [4]
    global_actor_critic = ActorCritic(input_dims, nb_actions)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [
        Agent(global_actor_critic, optim, input_dims, nb_actions, gamma=0.99, lr=lr, name=i,
              global_ep_index=global_ep,
              env_id=env_id) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers]