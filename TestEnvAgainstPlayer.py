import asyncio  
import numpy as np  
from poke_env import AccountConfiguration, LocalhostServerConfiguration  
from poke_env.player import Player  
import time
import pandas as pd
from stable_baselines3 import PPO
from ServerStart import ConnectAndOpen 
import matplotlib.pyplot as plt
ConnectAndOpen()
print("Please enter your user name as HumanPlayer_gen6 in browser")
time.sleep(15)
class PPOPlayer(Player):  
    def __init__(self, model_path, **kwargs):  
        super().__init__(**kwargs)  
        self.model = PPO.load(model_path)  
      
    def choose_move(self, battle):  
        # Convert battle to observation  
        obs = self.embed_battle(battle)  
          
        # Get action from trained model  
        action,_= self.model.predict(obs, deterministic=True)  
        if hasattr(action, 'item'):  
            action = int(action.item())  
        elif isinstance(action, np.ndarray):  
            action = int(action[0])  
        else:  
        # If it's already a scalar, just convert to int  
            action = int(action)  
        # Convert action to battle order  
        return self.action_to_order(action, battle)  
      
    def embed_battle(self, battle):  
        # We pass 14 dimensional vector using the same logic that we used during training  
        moves_base_power = -np.ones(4, dtype=np.float32)  
        moves_dmg_multiplier = np.ones(4, dtype=np.float32)  
          
        if battle.available_moves:  
            for i, move in enumerate(battle.available_moves[:4]):  
                moves_base_power[i] = (move.base_power or 0) / 100.0  
                  
                if battle.opponent_active_pokemon:  
                    moves_dmg_multiplier[i] = battle.opponent_active_pokemon.damage_multiplier(move)  
          
        fainted_mon_team = 0.0  
        fainted_mon_opponent = 0.0  
          
        if battle.team:  
            fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6.0  
            health_self= battle.active_pokemon.current_hp_fraction
            can_mega_self=1.0 if battle.can_mega_evolve else 0.0

        if battle.opponent_team:  
            fainted_mon_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6.0  
            health_opponent=battle.opponent_active_pokemon.current_hp_fraction
            can_mega_opponent=0.0 if not battle.opponent_used_mega_evolve else 1.0

        Observation = np.concatenate([  
            moves_base_power,  
            moves_dmg_multiplier,  
            [fainted_mon_team, fainted_mon_opponent],
            [health_self,health_opponent],
            [can_mega_self,can_mega_opponent]  
        ])  
          
        return np.nan_to_num(Observation, nan=0.0)  
      
    def action_to_order(self, action, battle):  
        # Convert action index to battle order    
        if action < 6:  # Switch action  
            available_switches = [p for p in battle.team.values() if not p.fainted and p != battle.active_pokemon]  
            if action < len(available_switches):  
                return self.create_order(available_switches[action])  
        else:  # Move action  
            move_idx = (action - 6) % 4  
            if move_idx < len(battle.available_moves):  
                return self.create_order(battle.available_moves[move_idx])  
          
        # Default to random move if action is invalid  
        return self.choose_random_move(battle)  
  
async def main():  
    # Load the trained agent  
    agent = PPOPlayer(  
        model_path="PPO_AGENT_DATA",  # Path to the saved model  
        account_configuration=AccountConfiguration("MyPPOBot", None),  
        server_configuration=LocalhostServerConfiguration,  
        battle_format="gen6randombattle"  
    )  
      
    # Wait for login  
    await agent.ps_client.logged_in.wait()  
    print(f"Agent '{agent.username}' logged in successfully!")  
    print("Ready to send challenges...")  
      
    # Send challenges to "Human Player"   
    await agent.send_challenges(  
        opponent="HumanPlayer_gen6",  
        n_challenges=1  # Number of battles to play  
    )  
    print("All battles completed successfully")  

#_______________________________________________________________________________________________________________________________________________________________    
    ## Log data into the log file
    try:
        df=pd.read_csv("WinRateVSHuman.csv")
    except FileNotFoundError:
        print("File doesn't exist\n Creating a new log file")
        df = pd.DataFrame(columns=["Battle_no","win_rate"])
    old_battles=len(df)
    total_battles_in_current_cycle=agent.n_finished_battles
    wins_in_current_cycle=agent.n_won_battles
    for match in range(total_battles_in_current_cycle):
        old_battles+=1
        total_wins=(df["win_rate"].iloc[-1]*(len(df))/100 if len(df)>0 else 0)+(1 if match<wins_in_current_cycle else 0) ## Calculate the number of battles played till now
        win_rate=(total_wins/old_battles) * 100 ## Finding winrate using total wins/total battles * 100
    df.loc[len(df)] = [old_battles, win_rate]
    df.to_csv("WinRateVSHuman.csv",index=False)
    ## Save data into WinRateVSHuman.csv

#__________________________________________________________________________________________________________________________________________________________________
    df = df.dropna(subset=["win_rate"]) ## Remove NULL values before plotting
    plt.figure(figsize=(8, 5))
    plt.plot(df["Battle_no"], df["win_rate"], linestyle="-",color="red")
    plt.title("Overall Win Rate Progress vs Human Player")
    plt.xlabel("Number of battles")
    plt.ylabel("Overall Win Rate (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("WinRateProgress.png", dpi=300)
    plt.close()
if __name__ == "__main__":  
    asyncio.run(main())