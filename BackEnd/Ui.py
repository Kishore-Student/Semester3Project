## Importing the modules for the UI

import sys
import os
import io
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QTextEdit, QPushButton, QTabWidget, QSizePolicy, QSplitter
)
from PySide6.QtCore import QThread, Signal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.image import imread
from TestEnvAgainstPlayer import TestVSHuman    ## Import the TestVSHuman function from the module
from RandomBattle import trainAgent         ## Import trainAgent function from the RandomBattle
import pandas as pd
# ------------------ WORKERS ------------------

class TrainWorker(QThread):
    log_signal = Signal(str)    ## Object to give signal to print text in GUI

    def run(self):
        sys.path.append(os.path.dirname(__file__))
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()   ## A temporary object where console outputs are written. (Lives in RAM)

        self.log_signal.emit("Starting training...")

        try:
            trainAgent()     ## Call the function
        except Exception as e:
            self.log_signal.emit(f"Error during training: {e}")
        finally:
            sys.stdout = old_stdout   ## Update the variable

        for line in mystdout.getvalue().splitlines():  ## Get the text to be printed
            self.log_signal.emit(line)      ## Forward the text to be printed to the function to print them

        self.log_signal.emit("Training finished!")

class BattleWorker(QThread):
    log_signal = Signal(str)     ## Create signal objects to call functions that display the log and plots 
    plot_signal = Signal(list)
    total_battles_signal = Signal(int)   

    def run(self):                          ## Defining the run method to allow updates to occur 
        def log(text):                      ## Function to send the text to the GUI object 
            self.log_signal.emit(text)     

        log("Starting battle/test...")
        TestVSHuman()                       ## Call the function that allows agent to battle against humans
        log("Battle/Test finished!")

        try:
            df = pd.read_csv("./Logs/WinRateVSHuman.csv")    ## Read the log file 
            self.plot_signal.emit(df["win_rate"].tolist())   ## Emit a signal to update the graph in canvas
            self.total_battles_signal.emit(df["Battle_no"].iloc[-1])   ## Get the total battle count from the log file
        except FileNotFoundError:
            log("WinRate CSV not found, skipping plot.")

# ------------------ MAIN WINDOW ------------------

class MainApp(QMainWindow):         ## Defining the main window class which would contain all the other elements
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Project (Pokemon showdown Bot)")     ## The name of the APP Window
        self.setGeometry(100, 100, 1200, 900)    ## Setting dimensions of the window : 
                                                ##Syntax as (x coordinate, y coordinate, width, height)
        central_widget = QWidget()             ## The container to contain other objects
        self.setCentralWidget(central_widget)   
        main_layout = QVBoxLayout()             ## Use automatic vertical spacer to arrange objects vertically
        central_widget.setLayout(main_layout)

        # Mode selection
        mode_layout = QHBoxLayout()             ## Arrange drop down list and button horizontally
        self.mode_dropdown = QComboBox()        ## Create a drop down list
        self.mode_dropdown.addItems(["Train", "Battle/Test"]) 
        self.start_button = QPushButton("Start") 
        self.start_button.setFixedWidth(150)     
        mode_layout.addWidget(QLabel("Select Mode:")) 
        mode_layout.addWidget(self.mode_dropdown)   #Looks as : Select Mode : [Drop down] [Button]
        mode_layout.addWidget(self.start_button)
        mode_layout.addStretch()
        main_layout.addLayout(mode_layout)
        self.start_button.clicked.connect(self.start_process) ## Call the functions when clicked

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # --- Train Tab ---
        self.train_tab = QWidget()
        self.train_layout = QVBoxLayout()
        self.train_tab.setLayout(self.train_layout)

        splitter = QSplitter(Qt.Horizontal)   ## This object would have 2 children to show plots of reward and winrate
        self.reward_canvas = FigureCanvas(Figure())   
        self.reward_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.winrate_canvas = FigureCanvas(Figure())
        self.winrate_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(self.reward_canvas)
        splitter.addWidget(self.winrate_canvas)
        self.train_layout.addWidget(splitter)

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setPlaceholderText("Training log...")
        self.train_layout.addWidget(self.train_log)
        self.tabs.addTab(self.train_tab, "Train")

        # --- Battle/Test Tab ---
        self.battle_tab = QWidget()
        self.battle_layout = QVBoxLayout()
        self.battle_tab.setLayout(self.battle_layout)

        self.battle_canvas = FigureCanvas(Figure())
        self.battle_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.battle_layout.addWidget(self.battle_canvas)

        self.battle_log = QTextEdit()
        self.battle_log.setReadOnly(True)    ## Setting log to read only to prevent it from accepting any key strokes
        self.battle_log.setPlaceholderText("Battle log...")
        self.battle_layout.addWidget(self.battle_log)

        self.total_battles_label = QLabel("Total Battles: 0")
        self.battle_layout.addWidget(self.total_battles_label)

        self.tabs.addTab(self.battle_tab, "Battle/Test")

    # ------------------ METHODS ------------------

    def start_process(self):
        mode = self.mode_dropdown.currentText()

        if mode == "Train":
            # prevent double-start
            if hasattr(self, "train_worker") and self.train_worker.isRunning():   ## Make sure only one training instance runs at a time.
                self.update_train_log("Training already running.")
                return

            self.train_worker = TrainWorker()    ## Connect threads to methods
            self.train_worker.log_signal.connect(self.update_train_log)
            self.train_worker.start()   ## Start running the run() method

            # Display pre-saved images
            self.display_reward_image()
            self.display_winrate_image()

        else:
            if hasattr(self, "battle_worker") and self.battle_worker.isRunning():
                self.update_battle_log("Battle already running.")
                return

            self.battle_worker = BattleWorker()
            self.battle_worker.log_signal.connect(self.update_battle_log)
            self.battle_worker.plot_signal.connect(self.update_battle_plot)
            self.battle_worker.total_battles_signal.connect(self.update_total_battles)
            self.battle_worker.start()

    # --- Train tab updates ---
    def update_train_log(self, text):
        self.train_log.append(text)

    def display_reward_image(self):
        img_path = "./Plot/RewardPlot.png"
        self.reward_canvas.figure.clear()   
        ax = self.reward_canvas.figure.add_subplot(111) # No of Rows, No of Cols, Index of subplot
        if os.path.exists(img_path):
            img = imread(img_path)
            ax.imshow(img, aspect='auto')
            ax.axis("off")
            ax.set_title("Reward Plot")
            self.reward_canvas.figure.tight_layout()
        else:
            ax.text(0.5, 0.5, "RewardPlot.png not found", ha='center', va='center')
            ax.axis("off")
        self.reward_canvas.draw()

    def display_winrate_image(self):
        img_path = "./Plot/WinRateVSBot.png"
        self.winrate_canvas.figure.clear()
        ax = self.winrate_canvas.figure.add_subplot(111)
        if os.path.exists(img_path):
            img = imread(img_path)
            ax.imshow(img, aspect='auto')
            ax.axis("off")
            ax.set_title("Win Rate vs Random (over all)")
            self.winrate_canvas.figure.tight_layout()
        else:
            ax.text(0.5, 0.5, "WinRateVSBot.png not found", ha='center', va='center')
            ax.axis("off")
        self.winrate_canvas.draw()

    # --- Battle tab updates ---
    def update_battle_log(self, text):
        self.battle_log.append(text)

    def update_battle_plot(self, win_rates):
        self.battle_canvas.figure.clear()
        ax = self.battle_canvas.figure.add_subplot(111)
        ax.plot(range(1, len(win_rates)+1), win_rates, color='green')
        ax.set_title("Overall win Rate against Human")
        ax.set_xlabel("Battles")
        ax.set_ylabel("Win Rate (%)")
        ax.set_xticks(range(1, len(win_rates)+1))
        ax.grid(True)
        self.battle_canvas.draw()

    def update_total_battles(self, n):
        self.total_battles_label.setText(f"Total Battles: {n}")

    # --- Cleanup ---
    def closeEvent(self, event):   ## If any event is running close it to prevent memory leakage
        # stop threads safely
        if hasattr(self, "train_worker") and self.train_worker.isRunning():
            self.train_worker.quit()
            self.train_worker.wait()
        if hasattr(self, "battle_worker") and self.battle_worker.isRunning():
            self.battle_worker.quit()
            self.battle_worker.wait()
        event.accept()


# ------------------ MAIN ------------------

if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = MainApp()
    window.show()

    exit_code = app.exec()
    window.deleteLater()
    sys.exit(exit_code)
