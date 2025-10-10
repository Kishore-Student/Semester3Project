import numpy as np  
from gymnasium.spaces import Box   
from poke_env.environment import SinglesEnv 
from poke_env.battle import AbstractBattle
 
class BaseEnv(SinglesEnv):  ## Creating base environment to wrap showdown observation environment on
    def __init__(self, **kwargs):  
        super().__init__(battle_format="gen6randombattle",strict=False, **kwargs) 
        """setting strict = false to prevent environment from crashing when there is an abnormal input
        and to fall to default behaviour instead"""  
          
        ## 10-dimensional observation space
        ## SETTING BOUNDS FOR OBSERVATION SPACE 
        self.observation_spaces = {  
            agent: Box(  
                np.array([-1, -1, -1, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32),  
                np.array([3, 3, 3, 3, 4, 4, 4, 4, 1, 1], dtype=np.float32),  
                dtype=np.float32,  
            )  
            for agent in self.possible_agents  
        } 
        """The first 4 values in the array represent move damage,
         in worst case the move may not have any damage at all or maybe 
         unavailable, in best case the move may have a base power of 300
         represented as 3 to keep the environment in the same scale
         
         The next 4 values specify the damage multipliers and 0 represents that 
         the opponent is completely immune to the moves and 4 represents that 
         the opponent takes 4 times the base damage due to it being 4 times weak to  that 
         move.
         
         The next value specifies the fraction of Pokemon fainted in agent's 
         side, and the last value specifies the fraction of Pokemon fainted in
         the opponent's side.
         """ 
    ## Override calc_reward function
    #Reward and penalty function for enabling critic policy to judge
    def calc_reward(self, battle: AbstractBattle) -> float:  
        return self.reward_computing_helper(  
            battle,   
            fainted_value=5.0,   #Setting reward for fainting opponent's pokemon to 5
            hp_value=1.0,        #Setting reward for reducing opponent's health to 1 per 1%
            victory_value=50.0   #Setting reward for winning the match to 50
        )  
   # The penalties for losing pokemon, health and battle are symmetric (i.e:-5,-1,-50 respectively).
    
    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:  # This function returns an array
        # Initialize with default values  
        move_base_dmg = -np.ones(4, dtype=np.float32)  
        move_dmg_multiplier = np.ones(4, dtype=np.float32)  
          
         ## Access the available moves of the current pokemon on field 
        if battle.available_moves:  
            for i, move in enumerate(battle.available_moves[:4]):  ## there are 4 moves and hence access those 4 moves and update the damages of those
                move_base_dmg[i] = (move.base_power or 0) / 100.0  ## Fill default value as 0 if the value is -1
                  
             ## Check if opponent has sent their Pokemon or not and then update the multiplier  
                if battle.opponent_active_pokemon:  
                    move_dmg_multiplier[i] = battle.opponent_active_pokemon.damage_multiplier(move)  
          
        ## Set the  ratio of Pokemon fainted initially on both sides (n/6)
        fainted_self = 0.0  
        fainted_opponent = 0.0  
         #Update the values  
        if battle.team:  
            fainted_self = len([Pokemon for Pokemon in battle.team.values() if Pokemon.fainted]) / 6.0  
          
        if battle.opponent_team:  
            fainted_opponent = len([Pokemon for Pokemon in battle.opponent_team.values() if Pokemon.fainted]) / 6.0  
          
        # Combine into final Observation vector  
        Observation = np.concatenate([  
            move_base_dmg,  
            move_dmg_multiplier,  
            [fainted_self, fainted_opponent],  
        ])  
          
        ## Replacing invalid observations  
        Observation = np.nan_to_num(Observation, nan=0.0)  
        #Return the final observation states  
        return Observation  