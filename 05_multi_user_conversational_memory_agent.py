# pip install pandas

import os

from langchain_openai.chat_models import ChatOpenAI
import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage,SystemMessage 




 
model = ChatOpenAI(
    temperature=1,
    max_tokens=1000,
    model='gpt-4o'
)
 
cakes_df = pd.read_csv("Data/cakes_data.csv")
 

 
@tool
def get_cake_price(cake_name:str) -> int :
    """
    This function retrieves the price of a cake based on a given name.  
    It searches for a partial match between the input name and the available cake names.  
    If a matching cake is found, its price is returned.  
    If no match is detected, the function returns None.
    """

   
    match_df = cakes_df[cakes_df["Nom-du-gâteau"].str.contains("^" + cake_name, case=False)]
    
    if len(match_df) == 0 : 
        return None
    else:
        return match_df["Prix"].iloc[0]
    


@tool
def get_cake_sugar(cake_name:str) -> int :
    """
    This function retrieves the sugar content of a cake based on a given name.  
    It searches for a partial match between the input name and the available cake names.  
    If a matching cake is found, its sugar percentage is returned.  
    If no match is detected, the function returns None.
    """
    match_df = cakes_df[cakes_df["Nom-du-gâteau"].str.contains("^" + cake_name, case=False)]
    if len(match_df) == 0 : 
        return None
    else:
        return match_df["Taux-de-sucre-(%)"].iloc[0]
  
@tool
def get_preparation_time(cake_name:str) -> int :
    """
    This function retrieves the preparation time of a cake based on a given name.  
    It searches for a partial match between the input name and the available cake names.  
    If a matching cake is found, its preparation time (in hours) is returned.  
    If no match is detected, the function returns None.
    """
    match_df = cakes_df[cakes_df["Nom-du-gâteau"].str.contains("^" + cake_name, case=False)]
    if len(match_df) == 0 : 
        return None
    else:
        return match_df["Délai-préparation-(h)"].iloc[0]

@tool
def get_cake_names(n:int) -> int :
    """
    This function returns a list of available cake names, limited to a specified number.  
    It retrieves the names from the available data source.  
    The function takes an integer parameter representing the maximum number of cake names to return.  
    If the requested number exceeds the total available cakes, it returns all available names.
    """
    return cakes_df["Nom-du-gâteau"].sample(n=min(n, len(cakes_df))).tolist()

@tool
def get_cakes_by_sugar_threshold(threshold:float) -> int :
    """
    This function returns a list of cake names where the sugar content is below a specified threshold.  
    It filters the cakes based on the given sugar percentage and retrieves only those that meet the criteria.  
    The function takes a numerical parameter representing the maximum allowed sugar percentage.  
    If no cakes meet the condition, it returns an empty list.
    """
    return cakes_df[cakes_df["Taux-de-sucre-(%)"] <= threshold]["Nom-du-gâteau"].tolist()

@tool
def get_total_cakes_count() -> int :
    """
    This function returns the total number of available cakes.  
    It counts all unique cake names from the available data source.  
    The function takes no parameters and returns an integer representing the total number of cakes.  
    """
    return cakes_df["Nom-du-gâteau"].nunique()





tools = [
        get_cake_price,
        get_cake_sugar,
        get_cake_names,
        get_cakes_by_sugar_threshold,
        get_total_cakes_count,
        get_preparation_time
        ]

system_prompt = SystemMessage("""
    You are a professional chatbot specializing in answering questions about your company's cakes while also capable of addressing general inquiries.
    To ensure accuracy, you will rely solely on the available tools and not your own memory.
    For small talk and greetings, you will maintain a professional and courteous tone in your responses.
    """
)

checkpointer=MemorySaver()

cake_QnA_agent=create_react_agent(
                                model=model,  
                                tools=tools,  
                                state_modifier=system_prompt,  
                                debug=False,  
                                checkpointer=checkpointer 
)



import uuid

config_1 = {"configurable": {"thread_id": str(uuid.uuid4())}}
config_2 = {"configurable": {"thread_id": str(uuid.uuid4())}}
config_3 = {"configurable": {"thread_id": str(uuid.uuid4())}} 
 
def execute_prompt(user, config, prompt):
    inputs = {"messages":[("user",prompt)]}
    ai_response = cake_QnA_agent.invoke(inputs,config=config)
    print(f"\n{user}: {ai_response['messages'][-1].content}")

execute_prompt("USER 1", config_1, "My favorite cake is Clafoutis.")
execute_prompt("USER 2", config_2, "My favorite cake is Crumble aux pommes.")
execute_prompt("USER 3", config_3, "My favorite cake is Brownie.")


execute_prompt("USER 1", config_1, "What is the sugar content of my favorite cake?")
execute_prompt("USER 2", config_2, "What is the sugar content of my favorite cake?")
execute_prompt("USER 3", config_3, "What is the sugar content of my favorite cake?")
 
 
 
  