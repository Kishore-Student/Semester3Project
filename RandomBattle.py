from poke_env.player import RandomPlayer  
from poke_env.environment import SingleAgentWrapper
from BaseEnv import BaseEnv
from stable_baselines3 import PPO
from poke_env import AccountConfiguration 
import os

agent="RL_AGENT_PPO"
os.system("cls" if os.name == "nt" else "clear") 
RandomBattle=BaseEnv(account_configuration1=AccountConfiguration("PPO_Agent", None),account_configuration2=AccountConfiguration("Random_Agent",None))
opponent=RandomPlayer(battle_format="gen6randombattle")
FinalEnvironment=SingleAgentWrapper(RandomBattle,opponent)
PPOAgent = PPO("MlpPolicy", FinalEnvironment, verbose=1)
PPOAgent.learn(total_timesteps=100000)  
#Save the Policy DATA into a file  
PPOAgent.save("PPO_AGENT_DATA")    
FinalEnvironment.close()

