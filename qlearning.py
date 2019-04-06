from IPython.display import HTML
#HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/q2ZOEFAaaI0?showinfo=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')

import numpy as np
import random
import gym

#create the environment
env=gym.make("FrozenLake-v0")

#create the q table & initilize it
#know the action&state size

action_size=env.action_space.n
state_size=env.observation_space.n

#start q table with 0
qtable=np.zeros((state_size,action_size))
print(qtable)


#set the hyperparameter
total_episodes=15000
test_episodes=100
learning_rate=0.8
max_steps=99
gamma=0.95   #It quantifies how much importance we give for future rewards

#set Exploration parameters
epsilon=1.0              #exploration rate
max_epsilon = 1.0        #exploration rate at the start
min_epsilon = 0.01       #min exploration rate at the start
decay_rate=0.005         # Exponential decay rate for exploration prob


#set the qlearning algorithm
rewards=[]
#until learning stop
for episode in range(total_episodes):
    #reset the environment after each episode

    state=env.reset()
    step=0
    done=False
    total_rewards=0

    for step in range(max_steps):
        # first we choose the action randomly
        exp_exp_tradeoff = random.uniform(0, 1)

        #take the greatest value for q  if exp_exp_tradeoff > epsilon: this mean that we in the exploitation
        if exp_exp_tradeoff > epsilon:
            action=np.argmax(qtable[state,:])    #action=np.argmax(qtable[state,:]) all the action can we take for this state

        #ELSE this mean that we in the exploration. take random action
        else:
            action=env.action_space.sample()

        #take the action then see the outcome state & reward
        new_state,reward,done,info=env.step(action)

        #the update qtable
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state,action]=qtable[state,action]+learning_rate*(reward+gamma*np.max(qtable[new_state,:])-qtable[state,action])

        total_rewards+=reward
        #update state
        state=new_state

        #if we finish epsilons
        if done==True:
            break
    # Reduce epsilon (because we need less and less exploration)
    epsilon=min_epsilon+(max_epsilon-min_epsilon)*np.exp(-decay_rate*episode)

    reward.append(total_rewards)

print ("Score over time: " +  str(sum(rewards)/total_episodes))
print(qtable)


#use q learning to play a game
env.reset()

for epsilon in range(5):
    state=env.reset()
    step=0
    done=False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        # take the action that have max Q
      action = np.argmax(qtable[state,:])

    new_state,reward,done,info =env.step(action)

    if done:
        # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
        env.render()

        # We print the number of step it took.
        print("Number of steps", step)
        break
    state = new_state

env.close
