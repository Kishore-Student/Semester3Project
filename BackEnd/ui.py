import streamlit as st
import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit page setup
st.set_page_config(page_title="Pokémon Battle Dashboard", layout="wide")

# Sidebar controls
st.sidebar.title("⚙️ Controls")

st.title("🎮 Pokémon Showdown RL Dashboard")

# --- Train PPO Agent ---
if st.sidebar.button("🚀 Train PPO Agent (vs Random)"):
    st.info("Training PPO Agent... Please wait, this may take a few minutes ⏳")
    with st.spinner("Training PPO Agent vs RandomPlayer..."):
        result = subprocess.run(["python", "RandomBattle.py"], capture_output=True, text=True)
    st.success("✅ Training complete! Model saved and logs updated.")
    st.text(result.stdout)

# --- Test Against Human Player ---
if st.sidebar.button("🎯 Test Against Human Player"):
    st.warning("⚠️ Please log in manually in your browser as: HumanPlayer_gen6 before continuing.")
    with st.spinner("Testing PPO Agent against Human Player..."):
        result = subprocess.run(["python", "TestEnvAgainstPlayer.py"], capture_output=True, text=True)
    st.success("✅ Testing complete! Win rate updated and progress plotted.")
    st.text(result.stdout)

# --- Reward Plot Section ---
st.header("📊 Reward Plot (Training)")
reward_plot = "./Plot/RewardPlot.png"
if os.path.exists(reward_plot):
    st.image(reward_plot, caption="Reward Plot During Training", use_container_width=True)
else:
    st.info("Train the agent first to see the reward plot.")

# --- Win Rate Progress Section ---
st.header("📈 Win Rate Progress (vs Human Player)")
winrate_plot = "./Plot/WinRateProgress.png"
if os.path.exists(winrate_plot):
    st.image(winrate_plot, caption="Win Rate Progress vs Human Player", use_container_width=True)
else:
    st.info("Run the test against a human player to generate this plot.")

# --- Recent Logs Section ---
st.header("📜 Recent Win Rate Logs")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Training Logs (vs Random)")
    rand_log_path = "./Logs/WinRateVSRand.csv"
    if os.path.exists(rand_log_path):
        df = pd.read_csv(rand_log_path)
        st.dataframe(df.tail(), use_container_width=True)
    else:
        st.warning("No training logs found yet.")

with col2:
    st.subheader("Testing Logs (vs Human Player)")
    human_log_path = "./Logs/WinRateVSHuman.csv"
    if os.path.exists(human_log_path):
        df = pd.read_csv(human_log_path)
        st.dataframe(df.tail(), use_container_width=True)
    else:
        st.warning("No human test logs found yet.")
