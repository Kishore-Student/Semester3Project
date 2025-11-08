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
import os
import matplotlib
matplotlib.use('Agg')

def trainAgent():
       ConnectAndOpen() ## Connect to the server and then open the default browser to run the server on
       time.sleep(10)
       agent="RL_AGENT_PPO" 
       RandomBattle=BaseEnv(account_configuration1=AccountConfiguration("PPO_Agent", None),account_configuration2=AccountConfiguration("Random_Agent",None))
       opponent=RandomPlayer(battle_format="gen6randombattle")
       environment=SingleAgentWrapper(RandomBattle,opponent)
       os.makedirs("./Logs", exist_ok=True) ## Create a directory if it doesn't exist
       FinalEnvironment=Monitor(environment,"./Logs/Training_logs")
       PPOAgent = PPO("MlpPolicy",         ## Creation of the PPO Agent as an object
                     FinalEnvironment,
                     verbose=1,
              ) 
       PPOAgent.learn(total_timesteps=8000)   ## Environment allows n observations and moves per run
       #Save the Policy DATA into the data file
       os.makedirs("./AgentDATA", exist_ok=True) ## Create a directory if it doesn't exist
       PPOAgent.save("./AgentDATA/PPO_AGENT_DATA")
       print(f"""Win Rate of the PPO agent is {(FinalEnvironment.env.env.agent1.win_rate * 100):.2f}% 
              It has won {FinalEnvironment.env.env.agent1.n_won_battles} battles out of {FinalEnvironment.env.env.agent1.n_finished_battles} battles""") ## Print winrate of the PPO agent at the end
       FinalEnvironment.close()
       #______________________________________________________________________________________

       try:
              df = pd.read_csv("./Logs/WinRateVSRand.csv")
       except FileNotFoundError:
              print("File doesn't exist. Creating a new log file.")
              df = pd.DataFrame(columns=["Instance", "overall_win_rate", "won_battles", "n_battles"])

       # Current training cycle stats
       total_battles = FinalEnvironment.env.env.agent1.n_finished_battles
       wins = FinalEnvironment.env.env.agent1.n_won_battles
       prev_total_battles = df["n_battles"].iloc[-1] if len(df) > 0 else 0
       prev_total_wins = (df["overall_win_rate"].iloc[-1] / 100 * prev_total_battles) if len(df) > 0 else 0

       # Update cumulative stats
       new_total_battles = prev_total_battles + total_battles
       new_total_wins = prev_total_wins + wins
       win_rate = (new_total_wins / new_total_battles) * 100

       # Append to log
       df.loc[len(df)] = [new_total_battles, win_rate, int(new_total_wins), new_total_battles]
       df.to_csv("./Logs/WinRateVSRand.csv", index=False)

       #_____________________________________________________________________________________
       #Generate a plot based on rewards
       df=pd.read_csv("./Logs/Training_logs.monitor.csv",skiprows=1)
       os.makedirs("./Plot", exist_ok=True) ## Create a directory if it doesn't exist
       plt.figure(figsize=(8, 5))
       plt.plot(df.index+1, df["r"], linestyle="-",color="red")
       plt.title("Reward plot of agent during training")
       plt.xlabel("Number of instances")
       plt.ylabel("Reward")
       plt.grid(True)
       plt.tight_layout()
       plt.savefig("./Plot/RewardPlot.png", dpi=300, bbox_inches='tight', pad_inches=0)
       plt.close()
       #_____________________________________________________________________________________
       df=pd.read_csv("./Logs/WinRateVSRand.csv")
       plt.figure(figsize=(8, 5))
       plt.plot(df.index,df["overall_win_rate"],linestyle="-",color="blue")
       plt.xlabel("Number of instances")
       plt.ylabel("Win rate during training")
       plt.grid(True)
       plt.tight_layout()
       plt.savefig("./Plot/WinRateVSBot.png", dpi=300, bbox_inches='tight', pad_inches=0)
       plt.close()

       
if __name__ == "__main__":
    trainAgent()
# from poke_env.player import RandomPlayer  
# from poke_env.environment import SingleAgentWrapper
# from BaseEnv import BaseEnv
# from stable_baselines3 import PPO
# from poke_env import AccountConfiguration 
# from ServerStart import ConnectAndOpen
# from stable_baselines3.common.monitor import Monitor
# import matplotlib.pylab as plt
# import time
# import pandas as pd
# import os

# def get_unique_agent_name(base="PPO_Agent"):
#     """
#     Generate a unique agent name <=18 chars using a persistent counter.
#     """
#     os.makedirs("./Logs", exist_ok=True)
#     counter_file = "./Logs/agent_counter.txt"
    
#     if os.path.exists(counter_file):
#         with open(counter_file, "r") as f:
#             count = int(f.read().strip()) + 1
#     else:
#         count = 1
    
#     with open(counter_file, "w") as f:
#         f.write(str(count))
    
#     # Limit total name length to 18 chars
#     return f"{base[:10]}{count:03d}"  # e.g., PPO_Agent001

# def trainAgent():
#     ConnectAndOpen()  # Start server & open browser
#     time.sleep(10)

#     agent_name = get_unique_agent_name("PPO_Agent")
#     random_agent_name = get_unique_agent_name("Random_Agent")

#     # Environment setup
#     RandomBattle = BaseEnv(
#         account_configuration1=AccountConfiguration(agent_name, None),
#         account_configuration2=AccountConfiguration(random_agent_name, None)
#     )
#     opponent = RandomPlayer(battle_format="gen6randombattle")
#     environment = SingleAgentWrapper(RandomBattle, opponent)

#     os.makedirs("./Logs", exist_ok=True)
#     FinalEnvironment = Monitor(environment, "./Logs/Training_logs")

#     # PPO agent
#     PPOAgent = PPO(
#         "MlpPolicy",
#         FinalEnvironment,
#         verbose=1
#     )

#     PPOAgent.learn(total_timesteps=8000)

#     # Save model
#     os.makedirs("./AgentDATA", exist_ok=True)
#     PPOAgent.save("./AgentDATA/PPO_AGENT_DATA")

#     print(f"""Win Rate of the PPO agent ({agent_name}) is {(FinalEnvironment.env.env.agent1.win_rate * 100):.2f}% 
#           It has won {FinalEnvironment.env.env.agent1.n_won_battles} battles out of {FinalEnvironment.env.env.agent1.n_finished_battles} battles""")

#     FinalEnvironment.close()

#     # Update cumulative win rate CSV
#     try:
#         df = pd.read_csv("./Logs/WinRateVSRand.csv")
#     except FileNotFoundError:
#         df = pd.DataFrame(columns=["Instance", "overall_win_rate", "won_battles", "n_battles"])

#     total_battles = FinalEnvironment.env.env.agent1.n_finished_battles
#     wins = FinalEnvironment.env.env.agent1.n_won_battles
#     prev_total_battles = df["n_battles"].iloc[-1] if len(df) > 0 else 0
#     prev_total_wins = (df["overall_win_rate"].iloc[-1] / 100 * prev_total_battles) if len(df) > 0 else 0

#     new_total_battles = prev_total_battles + total_battles
#     new_total_wins = prev_total_wins + wins
#     win_rate = (new_total_wins / new_total_battles) * 100

#     df.loc[len(df)] = [new_total_battles, win_rate, int(new_total_wins), new_total_battles]
#     df.to_csv("./Logs/WinRateVSRand.csv", index=False)

#     # Generate reward plot
#     df_plot = pd.read_csv("./Logs/Training_logs.monitor.csv", skiprows=1)
#     os.makedirs("./Plot", exist_ok=True)
#     plt.figure(figsize=(8, 5))
#     plt.plot(df_plot.index + 1, df_plot["r"], linestyle="-", color="red")
#     plt.title("Reward plot of agent during training")
#     plt.xlabel("Number of instances")
#     plt.ylabel("Reward")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("./Plot/RewardPlot.png", dpi=300)
#     plt.close()

# if __name__ == "__main__":
#     trainAgent()

