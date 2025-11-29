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
## Third-Party Libraries

This project uses the following third-party libraries, which are subject to their own licenses:

- **PokeEnv** – MIT License ([Documentation](https://poke-env.readthedocs.io/en/stable/))
- **Stable-Baselines3 (SB3)** – MIT License ([Documentation](https://stable-baselines3.readthedocs.io/en/master/))
- **PySide6** – LGPL License ([Documentation](https://doc.qt.io/qtforpython/))


## Contributors

| Name       | GitHub Profile | Contributions |
|------------|----------------|---------------|
| Kishore    | [Kishore-Student](https://github.com/Kishore-Student) | RL agent setup and coding; project environment setup; UI design; plotting; logging |
| Manjunath  | [Manjunathar415](https://github.com/Manjunathar415) | UI design; documentation |


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.
