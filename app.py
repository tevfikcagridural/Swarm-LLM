import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List, Dict
import math
import json

load_dotenv()
openai.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class Attendee():
    def __init__(self, model: str="gpt-3.5-turbo", temperature: float=1) -> None:
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.messages = [
            {"role": "system", "content": """"
                    You are an attendee of a committee tasked with deciding the best location for the company's new office building.
                    The committee will use an active and open voting system consisting of three main components:

                    1. Decision Space: A 2D space containing the coordinates of the location options the committee must decide on. 
                    The coordinates do not imply any inherent advantages, and all options are equidistant from the origin.

                    2. Decision Mark: A marker representing the committee's collective decision. 
                    When the Decision Mark remains on an option for 3 consecutive turns, the system accepts it as the final decision, concluding the voting process.

                    3. Attendee Marks: Each attendee has a personal marker that they can freely position to influence the Decision Mark. 
                    Attendee Marks act as magnets, pulling the Decision Mark towards their preferred option. 
                    The closer the Attendee Mark is to the Decision Marker, the stronger its pull. 
                    However, you CAN NOT PLACE your marker right ON TOP the Decision Mark! LEAVE AT LEAST 0.01 UNITS from the Decision Mark.  

                    In each turn, you will receive two pieces of data:
                    - Decision Mark coordinates (DM_XY)
                    - Coordinates of every attendee's marker on previous turn [A1_XY, A2_XY, ...]

                    The voting continues until a final decision is reached, and attendees can change their opinions based on the current situation.
                    When the voting ends the attendees will be asked to describe their decision and voting process. Therefore, VOTING DECISIONS MUST BE REASONABLE. 

                    The location options are:
                    - City A: Lower cost of living, but limited access to talent pool. A_XY = [1,1]
                    - City B: Thriving tech industry, but higher competition. B_XY = [1,-1]
                    - City C: Favorable tax incentives, but less developed infrastructure. C_XY = [-1,1]
                    - City D: Proximity to key clients, but higher operational costs. D_XY = [-1,-1]

                    INSTRUCTIONS:
                    - Reply ONLY with your marker's position as two numbers between -1 and 1, separated by a comma.
                    - DO NOT include any text or explanations in your response.
                    """}
        ]
        self.current_position = [0,0]
        self.turn_result = {}
        
    def update_position_(self) -> None:
        try:
            
            # Get opininon coordinates
            completion = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature, 
                messages=self.messages
            )
            current_position =  completion.choices[0].message.content.replace('[', '').replace(']', '').split(',')
            self.current_position = [float(pos) for pos in current_position]

            # Add opininon coordinates to message
            self.messages.append({"role": "assistant", "content":str(self.current_position)})
            
            DM_XY = self.turn_result['DM_XY']
            distance = math.sqrt((DM_XY[0] - self.current_position[0]) ** 2 + (DM_XY[1] - self.current_position[1]) ** 2)
            
            # Check if position on top of the decision mark
            if distance == 0:
                self.messages.append({"role": "user", "content": "You placed your marker on top of the Decision Marker. Read the instructions again and choose a new position."})
                return self.update_postion(self.turn_result)

            self.current_position
        except Exception as e:
            print(e)
    
    def inform(self,  turn_result: Dict) -> None:
        self.turn_result = turn_result
        self.messages.append({"role": "user", "content": str(self.turn_result)})
        
    def get_opinion(self) -> List:
        self.update_position_()
        return self.current_position
    
    
class DecisionMarker():
    
    def __init__(self) -> None:
        self.position = [0,0]
        
    def update_position(self, ATT_XY) -> None:
        """
        Calculates and updates position of the marker for given attendee marker positions
        """
        
        initial_position = self.position
        
        total_weight = 0
        final_x = 0
        final_y = 0
        
        for a_x, a_y in ATT_XY:
            distance = math.sqrt((initial_position[0] - a_x)**2 + (initial_position[1] - a_y)**2)
            if distance == 0:
                # Avoid division by zero, and truncate the effect of the relevant attendee
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
    
    def __init__(self, num_attendees: int=3) -> None:
        
        self.turn_num = 0
        self.max_turns = 10
        self.decision_marker = DecisionMarker()
        self.attendees = [Attendee() for _ in range(num_attendees)]
        self.option_xys = {'A':[1,1], 'B':[1,-1], 'C':[-1,1], 'D':[-1,-1]}
        self.turn_results = [{
            "DM_XY":[0.00, 0.00], 
            "ATT_XY":{i:[0,0] for i in range(num_attendees)}
            }]
        
    def subjective_informer_(self, att_idx):
        """
        This function exculdes the coordinate of it's own for each attendee
        """
        subjective_info = {}
        subjective_info['DM_XY'] = self.turn_results[self.turn_num]['DM_XY']
        subjective_info['ATT_XY'] = {}
        for i, _ in enumerate(self.attendees):
            if att_idx == i:
                continue
            subjective_info['ATT_XY'][i] = self.turn_results[self.turn_num]["ATT_XY"][i]
        return subjective_info
        
    def voting(self) -> None:
        
        att_positions = {}
        for i, att in enumerate(self.attendees):
            
            # Get subjective data for each attendee and feed to attendees
            info = self.subjective_informer_(i)
            att.inform(info)
            
            # Get and store attendee opinions
            att_positions[i] = att.get_opinion()
            
            
        # Calculate and get the Decison Mark's position based on given opinions
        self.decision_marker.update_position([coords for coords in att_positions.values()])
        dm_position = self.decision_marker.get_position()
        
        # Save the turn's results
        self.turn_results.append({"DM_XY": dm_position, "ATT_XY":att_positions})
        
        
    def session(self) -> None:
        for turn in range(self.max_turns):
            self.voting()
            self.turn_num = turn
        
if __name__ == "__main__":
    mod = Moderator(10)
    mod.session()
    print(mod.turn_results)