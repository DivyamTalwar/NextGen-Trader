from typing_extensions import Annotated, Sequence, TypedDict

import operator
from langchain_core.messages import BaseMessage


import json

""" ***a and ***b means take everything from a and then everything from b and concat. no duplicates. Output will be dict only 
We are getting output from multiple agents so just concatenate them into one simple output"""
def merge_dicts(a: dict[str, any], b: dict[str, any]) -> dict[str, any]:
    return {**a, **b}


# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add] #will contain the list of messages as value 
    data: Annotated[dict[str, any], merge_dicts]
    metadata: Annotated[dict[str, any], merge_dicts]


""" This function is defined to print out the output nicely along with the AI Agent Name """
def show_agent_reasoning(output, agent_name):
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}") #prints the header with agent name in the middle(*** agentname ***(10)

    """IT takes any kind of object and returns it into a proper json format (so u can use json.dump without error)"""
    def convert_to_serializable(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        
        elif isinstance(obj, (list, tuple)):#this means if the obj is an instance of list,tuple convert to serializble 
            return [convert_to_serializable(item) for item in obj]
        
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        
        else:
            return str(obj)  # Fallback to string representation
        
    """If the output is an instance of dict,list then we have to convert it into Json friendly format using convert func"""
    if isinstance(output, (dict, list)):
        serializable_output = convert_to_serializable(output)
        print(json.dumps(serializable_output, indent=2)) #json.dumps: Converts a Python object to a JSON string.
    #Meaning If Its A String(already in json format)    
    else:
        try:
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError: #it json.dump gives error simple print the output
            print(output)

    print("=" * 48)
