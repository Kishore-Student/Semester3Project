import torch
torch.set_num_threads(1)      ## Cap the number of threads running at a time to prevent Out of memory errors
torch.backends.cudnn.enabled = False   ## GPU usage disabled

from poke_env.player import RandomPlayer  
from poke_env.environment import SingleAgentWrapper
from BaseEnv import BaseEnv
from stable_baselines3 import PPO
from poke_env import AccountConfiguration 
from ServerStart import ConnectAndOpen
from stable_baselines3.common.monitor import Monitor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
    

def generate_unique_name(base_name):    ## Function to generate unique names to avoid naming errors
    """Generate a unique name ≤20 chars using timestamp suffix."""
    suffix = str(int(time.time()))[-6:]  # get last 6 digits of timestamp and append it to name to avoid duplicate name errors
    name = f"{base_name}{suffix}"[:20]
    return name

def trainAgent():
    # Ensure server is running
    ConnectAndOpen()
    time.sleep(15)  # give Node some time
    print("Server runs at local host 8000, check using the link http://localhost:8000")
    # Generate unique names
    agent_name_ppo = generate_unique_name("PPOAgent")
    agent_name_random = generate_unique_name("RandomAgent")

    RandomBattle = BaseEnv(
        account_configuration1=AccountConfiguration(agent_name_ppo, None),
        account_configuration2=AccountConfiguration(agent_name_random, None)
    )
    opponent = RandomPlayer(battle_format="gen6randombattle")
    environment = SingleAgentWrapper(RandomBattle, opponent)
    os.makedirs("./Logs", exist_ok=True)
    FinalEnvironment = Monitor(environment, "./Logs/Training_logs")   ## Logging into the log file

    PPO_Path = './AgentDATA/PPO_AGENT_DATA.zip'

    # Load or create PPO agent
    try:
        print("Agent found, loading...")
        PPOAgent = PPO.load(PPO_Path, env=FinalEnvironment)
    except FileNotFoundError:
        print("No previous model, creating new PPO agent...")
        PPOAgent = PPO(
            "MlpPolicy",
            FinalEnvironment,
            verbose=1,
            n_steps=128,     # rollout
            batch_size=16,   #  batch
            n_epochs=3       #  epochs
        )

    PPOAgent.learn(total_timesteps=4000)  # training 

    os.makedirs("./AgentDATA", exist_ok=True)
    PPOAgent.save("./AgentDATA/PPO_AGENT_DATA")

    print(f"Training finished. Win rate: {(FinalEnvironment.env.env.agent1.win_rate * 100):.2f}%")

    # Save logs
    try:
        df = pd.read_csv("./Logs/WinRateVSRand.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Instance", "overall_win_rate", "won_battles", "n_battles"])

    total_battles = FinalEnvironment.env.env.agent1.n_finished_battles  ## Getting the number of battles played
    wins = FinalEnvironment.env.env.agent1.n_won_battles    ##Getting number of battles won
    prev_total_battles = df["n_battles"].iloc[-1] if len(df) > 0 else 0
    prev_total_wins = (df["overall_win_rate"].iloc[-1] / 100 * prev_total_battles) if len(df) > 0 else 0

    new_total_battles = prev_total_battles + total_battles
    new_total_wins = prev_total_wins + wins
    win_rate = (new_total_wins / new_total_battles) * 100

    df.loc[len(df)] = [new_total_battles, win_rate, int(new_total_wins), new_total_battles]
    df.to_csv("./Logs/WinRateVSRand.csv", index=False)

    ## Generate reward and win rate plots
    df_reward = pd.read_csv("./Logs/Training_logs.monitor.csv", skiprows=1)
    os.makedirs("./Plot", exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(df_reward.index+1, df_reward["r"], color="red")
    plt.title("Reward plot")
    plt.xlabel("Instances")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Plot/RewardPlot.png", dpi=300,bbox_inches='tight', pad_inches=0)
    plt.close()

    df_win = pd.read_csv("./Logs/WinRateVSRand.csv")
    plt.figure(figsize=(8,5))
    plt.plot(df_win.index, df_win["overall_win_rate"], color="blue")
    plt.xlabel("Instances")
    plt.ylabel("Win rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Plot/WinRateVSBot.png", dpi=300,bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    trainAgent()
