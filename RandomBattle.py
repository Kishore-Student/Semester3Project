from poke_env.player import RandomPlayer  
from poke_env.environment import SingleAgentWrapper
from BaseEnv import BaseEnv
from stable_baselines3 import PPO
from poke_env import AccountConfiguration 
from ServerStart import ConnectAndOpen
from stable_baselines3.common.monitor import Monitor
import matplotlib.pylab as plt
import time
import pandas as pd

ConnectAndOpen() ## Connect to the server and then open the default browser to run the server on
time.sleep(10)
agent="RL_AGENT_PPO" 
RandomBattle=BaseEnv(account_configuration1=AccountConfiguration("PPO_Agent", None),account_configuration2=AccountConfiguration("Random_Agent",None))
opponent=RandomPlayer(battle_format="gen6randombattle")
environment=SingleAgentWrapper(RandomBattle,opponent)
FinalEnvironment=Monitor(environment,"Training_logs")
PPOAgent = PPO("MlpPolicy",         ## Creation of the PPO Agent as an object
                FinalEnvironment,
                verbose=1,
            ) 
PPOAgent.learn(total_timesteps=8000)   ## Environment allows n observations and moves per run
#Save the Policy DATA into the data file
PPOAgent.save("PPO_AGENT_DATA")    
print(f"""Win Rate of the PPO agent is {(FinalEnvironment.env.env.agent1.win_rate * 100):.2f}% 
       It has won {FinalEnvironment.env.env.agent1.n_won_battles} battles out of {FinalEnvironment.env.env.agent1.n_finished_battles} battles""") ## Print winrate of the PPO agent at the end
FinalEnvironment.close()

#_____________________________________________________________________________________
#Generate a plot based on rewards
df=pd.read_csv("Training_logs.monitor.csv",skiprows=1)
plt.figure(figsize=(8, 5))
plt.plot(df.index+1, df["r"], linestyle="-",color="red")
plt.title("Reward plot of agent during training")
plt.xlabel("Number of instances")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("RewardPlot.png", dpi=300)
plt.close()
