#!/usr/bin/env python3

"""
https://aistudio.google.com/ "Get API key"
https://aistudio.google.com/apikey
Put key in .env file with format:
GEMINI_API_KEY="the key"
Use OpenAIServerModel due to API compatibility
[Select model](https://ai.google.dev/gemini-api/docs/models)
"""

#############################################
# Environment loading
#############################################
import dotenv
import os
g_dotenv_loaded = False
def getenv(variable: str) -> str:
    global g_dotenv_loaded
    if not g_dotenv_loaded:
        g_dotenv_loaded = True
        dotenv.load_dotenv()
    value = os.getenv(variable)
    return value

api_key = getenv("GEMINI_API_KEY")

if not api_key:
    raise Exception("GEMINI_API_KEY needs to be set in .env.")

#############################################
# Model connection
#############################################
from smolagents import OpenAIServerModel

#model_id="gemini-2.0-flash"
#model_id="gemini-2.0-flash-lite"
model_id="gemini-2.5-flash"
model = OpenAIServerModel(model_id=model_id,
                          api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                          api_key=api_key,
                          )

answer = model.generate(messages=[{
    "role": "user", 
    "content": "Which is warmer? Blue or red?"
}])

print(f"Model returned answer: {answer.content}")



