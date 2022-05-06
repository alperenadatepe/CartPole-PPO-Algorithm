from dataclasses import dataclass

from .view_manager import ViewManager
from environment.trainer import Trainer

@dataclass
class App:
    def __post_init__(self):
        view_manager = ViewManager()
        
        if view_manager.buttons["train_button"]:
            trainer = Trainer(environment_name = 'MountainCar-v0', 
                            batch_size = view_manager.inputs["batch_size"], 
                            num_of_epochs = view_manager.inputs["num_of_epochs"], 
                            num_of_games = view_manager.inputs["num_of_games"], 
                            learning_rate = view_manager.inputs["learning_rate"], 
                            memory_size = view_manager.inputs["memory_size"],
                            ma_period = view_manager.inputs["ma_period"]
                        )
            
            actor_network_summary = trainer.agent.actor.get_summary()
            critic_network_summary = trainer.agent.critic.get_summary()

            view_manager.set_model_properties(actor_network_summary, critic_network_summary)
            
            with view_manager.set_spinner():
                trainer.start_training()

                view_manager.visualise_score(trainer.score_history)
                view_manager.set_status(success=True, text="Le-gen-dary!")
                

