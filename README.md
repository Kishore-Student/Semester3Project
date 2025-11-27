# Pokemon RL Trainer

A reinforcement learning backend and GUI system designed to train an AI agent to play Pokémon battles on [Pokemon Showdown](https://pokemonshowdown.com/).  
This project leverages **PokeEnv** and **Stable-Baselines3 (SB3)** for training, while providing a **PySide-based GUI** for monitoring, controlling, and visualizing the training process.

## Features
- Train an RL agent to play Pokémon battles using **PokeEnv** + **SB3**.
- **PySide-based GUI** for real-time monitoring and control.
- Automatic **server setup** for Pokemon Showdown.
- Logging of training progress and automatic **plotting of performance metrics**.
- Supports easy configuration of agents, battles, and learning parameters.

## Tech Stack
- **Python 3.10+**
- **Reinforcement Learning**: Stable-Baselines3 (SB3)
- **Pokémon Environment**: PokeEnv
- **GUI**: PySide6
- **Logging & Visualization**: Matplotlib

## Getting Started

### Prerequisites
- Python 3.10+
- pip (Python package manager)
- Node.js / npm if running your own Pokemon Showdown server

### Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/Kishore-Student/Semester5Project.git
cd Semester5Project
pip install -r requirements.txt
cd BackEnd
```
Run the GUI to start and monitor training:
```bash
python Ui.py
```
## References

- [PokeEnv Documentation](https://poke-env.readthedocs.io/en/stable/) – Official docs for PokeEnv.
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/) – Official docs for SB3.
