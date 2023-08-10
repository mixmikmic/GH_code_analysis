import matplotlib.pyplot as plt
import pandas as pd
from gridWorldEnvironment import GridWorld

# creating gridworld environment
gw = GridWorld()

print("Actions: ", gw.actions)
print("States: ", gw.states)

pd.DataFrame(gw.transitions, columns = ["State", "Action", "Next State", "Reward"]).head()

print(gw.state_transition(1, "U"))
print(gw.state_transition(3, "L"))

gw.show_environment()
plt.show()

