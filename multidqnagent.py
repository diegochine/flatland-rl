import numpy as np
from rl.agents import DQNAgent
from rl.callbacks import TrainIntervalLogger, TrainEpisodeLogger, Visualizer, CallbackList
from tensorflow.keras.callbacks import History

import config as cfg
import logger as log


class MultiDQNAgent(DQNAgent):
    """Abstract base class for all implemented agents.
    Each agent interacts with the environment (as defined by the `Env` class) by first observing the
    state of the environment. Based on this observation the agent changes the environment by performing
    an action.
    Do not use this abstract base class directly but instead use one of the concrete agents implemented.
    Each agent realizes a reinforcement learning algorithm. Since all agents conform to the same
    interface, you can use them interchangeably.
    To implement your own agent, you have to implement the following methods:
    - `forward`
    - `backward`
    - `load_weights`
    - `save_weights`
    """

    def __init__(self, model, name='agent', nb_agents=1, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self._logger = log.setup_logger(name, f'{"logs/" + name + ".txt"}')
        self.nb_agents = nb_agents
        self.recent_observation = None
        self.recent_action = None

    def forward(self, observation):
        actions = {}
        for a in range(self.nb_agents):
            # Select an action.
            if observation[a] is not None:
                state = [observation[a]]
                q_values = self.compute_q_values(observation[a])
                if self.training:
                    action = self.policy.select_action(q_values=q_values)
                else:
                    action = self.test_policy.select_action(q_values=q_values)
            else:
                action = 0
            actions.update({a: action})

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = actions

        return actions

    def backward(self, all_rewards, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self._logger.info(f'Memory saved at STEP {self.step:3d}')
            for a in range(self.nb_agents):
                self.memory.append(self.recent_observation[a], self.recent_action[a], all_rewards[a], terminal,
                                   training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            self._logger.info(f'Training at STEP {self.step:3d}')
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            self._logger.info(f'Computing Q values')
            if self.enable_double_dqn:
                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                q_values = self.model.predict_on_batch(state1_batch)
                assert q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)

            targets = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size,))
            masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            self._logger.info(f'Training on minibatch of size {self.batch_size}')
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
            metrics = [metric for idx, metric in enumerate(metrics) if
                       idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

            if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
                self._logger.info(f'Updating target model at STEP {self.step:3d}')
                self.update_target_model_hard()

            return metrics

    def fit(self, env, nb_episodes, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=500):
        if not self.compiled:
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')

        self.training = True
        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        # Empty dictionary for all agent action
        self._logger.info("Starting Training")

        for episode in range(nb_episodes):
            self._logger.info(f'EPISODE {episode:2d}/{nb_episodes}')
            # Reset environment and get initial observations for all agents
            next_obs, info = env.reset()
            # env_renderer.reset()
            episode_reward = 0
            step = 0
            done = {'__all__': False}
            # Run episode
            while not done['__all__'] and step < nb_max_episode_steps:
                self._logger.info(f'STEP {step:3d}/{nb_max_episode_steps:3d}')
                callbacks.on_step_begin(step)
                # Chose an action for each agent in the environment
                obs = {o: self.processor.process_observation(next_obs[o])
                       for o in next_obs if next_obs[o] is not None}
                obs.update({o: None
                            for o in next_obs if next_obs[o] is None})
                action_dict = self.forward(obs)

                callbacks.on_action_begin(action_dict)
                # Environment step which returns the observations for all agents, their corresponding
                # reward and whether their are done
                next_obs, all_rewards, done, info = env.step(action_dict)
                # env_renderer.render_env(show=True, show_observations=True, show_predictions=False)7
                callbacks.on_action_end(action_dict)

                # Update replay buffer and train agent
                metrics = self.backward(all_rewards, done)
                episode_reward += sum(all_rewards.values())
                step += 1
                step_logs = {
                    'action': action_dict,
                    'observation': obs,
                    'reward': episode_reward,
                    'metrics': metrics,
                    'episode': step,
                    'info': {}
                }
                callbacks.on_step_end(step, step_logs)
                self.step += 1

            # This episode is finished, report and reset.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_episode_steps': step,
                'nb_steps': self.step,
            }
            callbacks.on_episode_end(episode, episode_logs)

            episode += 1
            observation = None
            episode_step = None
            episode_reward = None

        callbacks.on_train_end(logs={'did_abort': False})
        self._on_train_end()

        return history
