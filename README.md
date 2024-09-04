# POLICY EVALUATION

## AIM
Write the experiment AIM.

## PROBLEM STATEMENT
♣ we are tasked with creating an RL agent to solve the "Bandit Slippery Walk" problem. ♣ The environment consists of Seven states representing discrete positions the agent can occupy. ♣ The agent must learn to navigate this environment while dealing with the challenge of slippery terrain. ♣ Slippery terrain introduces stochasticity in the agent's actions, making it difficult to predict the outcomes of its actions accurately.

## States
The environment has 7 states:

Two Terminal States: G: The goal state & H: A hole state.

Five Transition states / Non-terminal States including S: The starting state.

## Actions
The agent can take two actions:

R -> Move right.
L -> Move left.

## Transition Probabilities
The transition probabilities for each action are as follows:

50% chance that the agent moves in the intended direction.
33.33% chance that the agent stays in its current state.
16.66% chance that the agent moves in the opposite direction.

## Reward
♣ The agent receives a reward of +1 for reaching the goal state (G).

♣ The agent receives a reward of 0 for all other states.

## Graphical Representation:
![image](https://github.com/user-attachments/assets/ec9177b8-6120-4b62-a1e1-08655774def9)


## POLICY EVALUATION FUNCTION
```from numpy.lib.function_base import copy
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    while True:
      V=np.zeros(len(P),dtype=np.float64)
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s]+=prob*(reward+gamma+prev_V[next_state]*(not done))
      if np.max(np.abs(prev_V-V))<theta:
        break
      prev_V=V.copy()
      return V
# Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P,gamma=0.89)
print_state_value_function(V1, P, n_cols=7, prec=5)

# Code to evaluate the second policy
V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)

V1

print_state_value_function(V1, P, n_cols=7, prec=5)

V2

print_state_value_function(V2, P, n_cols=7, prec=5)

V1>=V2

if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
``` 
## OUTPUT:
### Policy 1 and state value function
![image](https://github.com/user-attachments/assets/90fefe80-731d-4e98-8e9a-9961e2139c96)


### Policy 2 and state value function
![image](https://github.com/user-attachments/assets/6af7c95c-f9df-4aa6-9318-ab04f268bd70)

### Comparision and conclusion 
![image](https://github.com/user-attachments/assets/4ae3ef0b-d9c5-497f-be0d-044002c0bf95)

## RESULT:

Thus the Given Policy have been Evaluated and Optimal Policy has been Computed using Python Programming.

