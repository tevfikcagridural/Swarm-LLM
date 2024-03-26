import os
from dotenv import load_dotenv
from openai import OpenAI, AssistantEventHandler
import math
from typing import Dict, List
import time

load_dotenv()

class Member():
        
    def __init__(self) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
        ## Reserved for tool usage
        # self.tools = [
        #     {
        #         "type": "function",
        #         "function": {
        #         "name": "get_current_weather",
        #         "description": "Get the current weather in a given location",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #             "location": {
        #                 "type": "string",
        #                 "description": "The city and state, e.g. San Francisco, CA",
        #             },
        #             "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        #             },
        #             "required": ["location"],
        #         },
        #         }
        #     }
        #     ]
        self.thread = self.client.beta.threads.create()
        self.current_position = [0,0]
        self.turn_result = {}
        
    def inform(self, turn_result: Dict) -> None:
        self.turn_result = turn_result
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=str(turn_result)
        )
        
    def update_position_(self) -> None:
        """
        Updates the position of the member based on the member's voting choice
        """
        
        # Get opinion coordinates
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant_id,
        ) 
        
        # Retrive the answer from assistant
        while run.status in ["queued", "in_progress"]:
            keep_retrieving_run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=run.id
            )
            
            if keep_retrieving_run.status == "completed":
                # Retrieve the messages added by the assistant to the thread
                all_messages = self.client.beta.threads.messages.list(
                    thread_id=self.thread.id
                )
                
                # Update the position with current decision
                current_position = all_messages.data[0].content[0].text.value
                self.current_position = [float(x) for x in current_position.split(",")]
                
                break
            elif keep_retrieving_run.status in ["queued", "in_progress"]:
                # Delay before the next retrieval attempt
                time.sleep(10)
                pass
            else:
                print(f"Unknown run status: {run.status}")
                break
            
    def vote(self) -> List:
        self.update_position_()
        return self.current_position
    

class DecisionMarker():
    
    def __init__(self) -> None:
        self.position = [0,0]
        
    def update_position(self, ATT_XY) -> None:
        """
        Calculates and updates position of the marker for given member marker positions
        """
        
        initial_position = self.position
        
        total_weight = 0
        final_x = 0
        final_y = 0
        
        for a_x, a_y in ATT_XY:
            distance = math.sqrt((initial_position[0] - a_x)**2 + (initial_position[1] - a_y)**2)
            if distance == 0:
                # Avoid division by zero, and truncate the effect of the relevant member
                continue
             
            weight = 1 / distance ** 2
            total_weight += weight 
            final_x += a_x * weight
            final_y += a_y * weight
            
        final_x /= total_weight
        final_y /= total_weight
        
        self.position = [final_x, final_y]
        
    def get_position(self) -> List:
        return self.position
    
    
class Moderator():
    """
    The Moderator class represents the moderator in a decision-making process. 
    It manages the members, facilitates voting, and updates the decision marker's 
    position based on the members' votes.
    """
    
    def __init__(self, num_members: int = 3, num_turns: int =10) -> None:
        """
        Parameters:

            num_members (int, optional): The number of members in the decision-making process. Defaults to 3.
            num_turns (int, optional): The number of turns in the decision-making process. Defaults to 10.
        
        Attributes:
        
            turn_num: The current turn number.
            max_turns: The maximum number of turns in the decision-making process.
            decision_marker: An instance of the DecisionMarker class
            members: A list of Member instances, representing the members in the decision-making process.
            turn_results: A list of dictionaries, where each dictionary represents the results of a single turn. Each dictionary contains the following keys:
                DM_XY: A list of coordinates representing the decision marker's position.
                ATT_XY: A dictionary of coordinates, where each key is a member index and the corresponding value is a list of coordinates representing the member's position.
        """
        self.turn_num = 0
        self.max_turns = 10
        self.decision_marker = DecisionMarker()
        self.members = [Member() for _ in range(num_members)]
        self.option_xys = {'A':[1,1], 'B':[1,-1], 'C':[-1,1], 'D':[-1,-1]}
        self.turn_results = [{
            "DM_XY":[0.00, 0.00], 
            "ATT_XY":{i:[0,0] for i in range(num_members)}
            }]
        
    def subjective_informer_(self, att_idx):
        """
        Excludes the member's own coordinate from the information provided to self.
        """
        subjective_info = {}
        subjective_info['DM_XY'] = self.turn_results[self.turn_num]['DM_XY']
        subjective_info['ATT_XY'] = {}
        for i, _ in enumerate(self.members):
            if att_idx == i:
                continue
            subjective_info['ATT_XY'][i] = self.turn_results[self.turn_num]["ATT_XY"][i]
        return subjective_info
        
    def voting(self) -> None:
        """
        Facilitates the voting process by informing each member of the subjective data and collecting their votes.
        """
        member_positions = {}
        for i, member in enumerate(self.members):
            
            # Get subjective data for each member and feed to members member
            info = self.subjective_informer_(i)
            member.inform(info)
            
            # Get and store member opinions
            member_positions[i] = member.vote()
            
            
        # Calculate and get the Decison Mark's position based on given opinions
        self.decision_marker.update_position([coords for coords in member_positions.values()])
        dm_position = self.decision_marker.get_position()
        
        # Save the turn's results
        self.turn_results.append({"DM_XY": dm_position, "ATT_XY":member_positions})
        
        
    def session(self) -> None:
        """
        Runs the decision-making process for the specified number of turns.
        """
        for turn in range(self.max_turns):
            self.voting()
            self.turn_num = turn
        
if __name__ == "__main__":
    mod = Moderator(num_members=3, num_turns=10)
    mod.session()
    print(mod.turn_results)