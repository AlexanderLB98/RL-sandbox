import pandas as pd

from src.base_agent import BaseAgent


class sarsa(BaseAgent):
    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.5,
        epsilon=0.9,
        epsilon_decay=0.9,
        epsilon_min=0.05,
        policy_strategy="epsilon_greedy",
    ) -> None:
        # Call the constructor of the parent class (BaseAgent)
        super().__init__(
            env=env,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            policy_strategy=policy_strategy,
        )

    def train(self, n_episodes=200):
        # Implement your training method here
        env = self.env
        
        self.Q = {}
        for s in range(env.observation_space.n):
            self.Q[s] = {}
            for a in range(env.action_space.n):
                self.Q[s][a] = 0
        print(self.Q)

        rewards = {}

        columns = ["episode", "step", "state", "action", "reward", "next_state", "next_action", "alpha", "epsilon", "gamma"]
        history = pd.DataFrame(columns=columns)

        self.initialize_plot()

        # Crear política epsilon-greedy
        self.policy = self.make_policy(self.Q, self.epsilon, env.action_space.n, method=self.policy_strategy)

        # 2. Loop por episodio
        for episode in range(n_episodes):
            i = 0 # counter
            obs = env.reset()[0]
            done = False
            rewards[episode] = 0
            episode_df = []

            # 3. Loop por step
            while not done:
                # action = epsilon_greedy(q_values=Q, epsilon=epsilon, n_actions=env.action_space.n)
                action1 = self.policy(state=obs)
                new_obs, reward, terminated, truncated, info = env.step(action1)
                done = terminated or truncated
                if not done:
                    action2 = self.policy(state=new_obs)
                    q_next = self.Q[new_obs][action2]
                else:
                    q_next = 0


                rewards[episode] += reward
                i += 1

                episode_df.append([episode, i, obs, action1, reward, new_obs, action2, self.alpha, self.epsilon, self.gamma])
                # history.append([obs, action1, reward, new_obs, action2]) # SARSA
                self.Q[obs][action1] = self.Q[obs][action1] + self.alpha * (reward + self.gamma * q_next - self.Q[obs][action1])

                # Update policy every timestep (on policy)
                self.policy = self.make_policy(self.Q, self.epsilon, env.action_space.n, method=self.policy_strategy)

                obs = new_obs
                action1 = action2




            self.episode_df = pd.DataFrame(episode_df, columns=columns)
            self.history = pd.concat([self.history, self.episode_df], ignore_index=True)
            
            print(self.Q)
            print(f"Episodio {episode} completado en {i} pasos con recompensa {rewards[episode]}.")
            print(f"Last 5 actions:  { list(history['action'].iloc[-5:])}")
            print(f"Q[0]: {self.Q[0]}")
            self.print_policy(self.policy, 4,4)
            self.update_plot()
            # Update epsilon for next episode
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


        return self.Q, self.policy, history


    def plot_training(self):
        return super().plot_training()




