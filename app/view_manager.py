from dataclasses import dataclass
import requests

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from utils.utils import calculate_ma

@dataclass
class ViewManager:
    def _inputs_area_init(self):
        st.sidebar.header("Input Values")
        st.sidebar.write("Select the settings and parameter values: ")

        environment_settings = st.sidebar.expander("Environment Settings")
        with environment_settings:
            self.inputs["num_of_games"] = st.slider('Number of Simulations', 10, 1000, 300, key = "num_of_games")
            self.inputs["ma_period"] = st.slider('MA Period for Visualisations', 10, 100, 30, key = "ma_period")

        hyper_parameters = st.sidebar.expander("Hyperparameters Settings")
        with hyper_parameters:  
            self.inputs["batch_size"] = st.slider('Batch Size', 1, 50, 5, key = "batch_size")
            self.inputs["memory_size"] = st.slider('Memory Size', 1, 50, 20, key = "memory_size")
            self.inputs["num_of_epochs"] = st.slider('Number of Epochs', 1, 50, 4, key = "num_of_epochs")
            self.inputs["learning_rate"] = st.number_input('Learning Rate', 0.0001, 1., 0.0003, 0.0001, format="%.4f", key = "learning_rate")
    
    def _train_button_init(self):
        train_button = st.button("Train Model")
        self.buttons["train_button"] = train_button
    
    def _download_button_init(self):
        col_1, col_2 = st.columns(2)
        with open("checkpoints/actor_torch_ppo.pt", "rb") as file:
            download_button = col_1.download_button(
                    label="Download Actor Model",
                    data=file,
                    file_name="actor_torch_ppo.pt"
                )
            self.buttons["download_actor_button"] = download_button
        
        with open("checkpoints/critic_torch_ppo.pt", "rb") as file:
            download_critic_button = col_2.download_button(
                    label="Download Critic Model",
                    data=file,
                    file_name="critic_torch_ppo.pt"
                )
            self.buttons["download_critic_button"] = download_critic_button

    def _streamlit_initialization(self):
        self.inputs, self.buttons = {}, {}
    
        st.write("""
        # Cart Pole Optimization with PPO Algorithm
        """)

        im = Image.open(requests.get("https://thumbs.gfycat.com/WelllitLawfulCero-mobile.jpg", stream=True).raw)
        st.image(im)

        st.markdown("""
        With supervised learning, we can easily implement the cost function, run gradient descent on it, and be very confident that we’ll get excellent results with relatively little hyperparameter tuning. The route to success in reinforcement learning isn’t as obvious — the algorithms have many moving parts that are hard to debug, and they require substantial effort in tuning in order to get good results. PPO strikes a balance between ease of implementation, sample complexity, and ease of tuning, trying to compute an update at each step that minimizes the cost function while ensuring the deviation from the previous policy is relatively small. [[OpenAI](https://openai.com/blog/openai-baselines-ppo/#ppo)]
        """)

        st.markdown("""<hr style="height:0.1px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        self._train_button_init()
        self._download_button_init()
        st.markdown("""<hr style="height:0.1px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


        st.markdown("""<hr style="height:0.1px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


        self._inputs_area_init()

    def __post_init__(self):
        self._streamlit_initialization()

    def set_spinner(self):
        return st.spinner('Wait for it...')
    
    def set_status(self, success, text):
        if success:
            st.success(text)
            st.balloons()

    def set_model_properties(self, actor_network_summary, critic_network_summary):
        col_1, col_2 = st.columns(2)

        col_1.title("Actor Network")
        col_1.text(actor_network_summary)

        col_2.title("Critic Network")
        col_2.text(critic_network_summary)

        st.markdown("""<hr style="height:0.1px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    def visualise_score(self, score_history):
        moving_average = calculate_ma(score_history, self.inputs["ma_period"])
        x_axis = [game_no + 1 for game_no in range(len(score_history))]
        
        fig, ax = plt.subplots()
        ax.plot(x_axis, moving_average)
        ax.set_title(f'Running average of previous {self.inputs["ma_period"]} scores')

        st.pyplot(fig)