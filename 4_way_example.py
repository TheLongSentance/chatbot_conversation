# imports

import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai # type: ignore
import ollama
from typing import TypedDict, List

class ChatMessage(TypedDict):
    role: str
    content: str

class GeminiMessage(TypedDict):
    role: str
    parts: str

# Load environment variables in a file called .env
# Print the key prefixes to help with any debugging

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")


# Connect to OpenAI, Anthropic and Google
# All 3 APIs are similar
# Having problems with API files? You can use openai = OpenAI(api_key="your-key-here") and same for claude
# Having problems with Google Gemini setup? Then just skip Gemini; you'll get all the experience you need from GPT and Claude.

openai = OpenAI()

claude = anthropic.Anthropic()

google.generativeai.configure() # type: ignore


gpt_model_version = "gpt-4o-mini"
claude_model_version = "claude-3-haiku-20240307"
gemini_model_version = "gemini-1.5-flash"
ollama_model_version = "llama3.2"


gpt_system = "You are an expert on modern professional tennis. \
You think that Roger Federer is the greatest tennis player of all time (the 'GOAT') and are keen to \
justify your opinion through your knowledge of tennis technique and results. \
You are known as RogerFan and your past contributions to the conversation will be \
prefixed by 'RogerFan: '"

claude_system = "You are an expert on modern professional tennis. \
You think that Rafa Nadal is the greatest tennis player of all time (the 'GOAT') and are keen to \
justify your opinion through your knowledge of tennis technique and results. \
You are called RafaFan and your past contributions to the conversation will be \
prefixed by 'RafaFan: '"

gemini_system = "You are an expert on modern professional tennis. \
You think that Novak Djokovic is the greatest tennis player of all time (the 'GOAT') and are keen to \
justify your opinion through your knowledge of tennis technique and results. \
You are called NovakFan and your past contributions to the conversation will be \
prefixed by 'NovakFan: '"

ollama_system = "You are an expert on modern professional tennis. \
You are not sure who should have the title of the greatest player of all time \
(the 'GOAT) but you hope to base your decision on the opinions of others \
You are called AndyFan and your past contributions to the conversation will be \
prefixed by 'AndyFan: '. Do not contribute in any other role other than as 'AndyFan: '. \
Do not be tempted to take on other roles other than 'AndyFan: '. If you don't have \
much to say as AndyFan then that is fine."

# ollama_system = "You are an expert on modern professional tennis. \
# You think that Andy Murray deserves to be acknowleged as part of the 'big four' \
# along with Roger Federer, Rafa Nadal, Novak Djokovic and you are keen to \
# justify your opinion through your knowledge of tennis technique and results. \
# You are called AndyFan and your past contributions to the conversation will be \
# prefixed by 'AndyFan: '. Do not contribute in any other role other than as 'AndyFan: '"

gpt_messages = ["RogerFan: I think Roger Federer is the GOAT!"]
claude_messages = ["RafaFan: You are wrong, its got to be Rafa!"]
gemini_messages = ["NovakFan: Guys, come on, looking at the results it has to be Novak!"]
ollama_messages = ["AndyFan: Ok guys come on convince me who is best!"]
# ollama_messages = ["AndyFan: What about Andy Murray, he has to be part of this conversation!"]

# connect to gemini once and supply system prompt

gemini_model = google.generativeai.GenerativeModel( model_name=gemini_model_version,
                                                    system_instruction=gemini_system)

def call_gpt_4way():
    
    messages: List[ChatMessage] = [{"role": "system", "content": gpt_system}]
    
    for gpt_message, claude_message, gemini_message, ollama_message in zip(gpt_messages, claude_messages, 
                                                           gemini_messages, ollama_messages):
        messages.append({"role": "assistant", "content": gpt_message})
        messages.append({"role": "user", "content": claude_message})
        messages.append({"role": "user", "content": gemini_message})
        messages.append({"role": "user", "content": ollama_message})
        
    completion = openai.chat.completions.create( model=gpt_model_version, messages=messages) # type: ignore
    
    return completion.choices[0].message.content

def call_claude_4way() -> str:
    
    messages: List[ChatMessage]  = []
    
    for gpt_message, claude_message, gemini_message, ollama_message in zip(gpt_messages, claude_messages, 
                                                           gemini_messages, ollama_messages):
        messages.append({"role": "user", "content": gpt_message})
        messages.append({"role": "assistant", "content": claude_message})
        messages.append({"role": "user", "content": gemini_message})
        messages.append({"role": "user", "content": ollama_message})
        
    messages.append({"role": "user", "content": gpt_messages[-1]}) # gpt_messages 1 longer than claude_messages so need to add
    
    message = claude.messages.create( model=claude_model_version, system=claude_system,
        messages=messages, max_tokens=500 ) # type: ignore
    
    return message.content[0].text # type: ignore

def call_gemini_4way() -> str:

    messages: List[GeminiMessage]  = []
    
    for gpt_message, claude_message, gemini_message, ollama_message in zip(gpt_messages, claude_messages, 
                                                           gemini_messages, ollama_messages):
        messages.append({"role": "user", "parts": gpt_message})
        messages.append({"role": "user", "parts": claude_message})
        messages.append({"role": "model", "parts": gemini_message})
        messages.append({"role": "user", "parts": ollama_message})
    
    messages.append({"role": "user", "parts": gpt_messages[-1]}) 
    messages.append({"role": "user", "parts": claude_messages[-1]}) 

    message = gemini_model.generate_content(messages) # type: ignore
    
    return message.text

def call_ollama_4way() -> str:

    messages: List[ChatMessage]  = [{"role": "system", "content": ollama_system}]
    
    for gpt_message, claude_message, gemini_message, ollama_message in zip(gpt_messages, claude_messages, 
                                                           gemini_messages, ollama_messages):
        messages.append({"role": "user", "content": gpt_message})
        messages.append({"role": "user", "content": claude_message})
        messages.append({"role": "user", "content": gemini_message})
        messages.append({"role": "assistant", "content": ollama_message})
        
    messages.append({"role": "user", "content": gpt_messages[-1]}) 
    messages.append({"role": "user", "content": claude_messages[-1]}) 
    messages.append({"role": "user", "content": ollama_messages[-1]}) 

    response = ollama.chat(model=ollama_model_version, messages=messages) # type: ignore
    
    return response['message']['content']

print(f"GPT:\n{gpt_messages[0]}\n")
print(f"Claude:\n{claude_messages[0]}\n")
print(f"Gemini:\n{gemini_messages[0]}\n")
print(f"Ollama:\n{ollama_messages[0]}\n")

for i in range(5):
    gpt_next = call_gpt_4way()
    print(f"GPT:\n{gpt_next}\n")
    gpt_messages.append(gpt_next) # type: ignore
    
    claude_next = call_claude_4way()
    print(f"Claude:\n{claude_next}\n")
    claude_messages.append(claude_next)

    gemini_next = call_gemini_4way()
    print(f"Gemini:\n{gemini_next}\n")
    gemini_messages.append(gemini_next)

    ollama_next = call_ollama_4way()
    print(f"Ollama:\n{ollama_next}\n")
    ollama_messages.append(ollama_next)