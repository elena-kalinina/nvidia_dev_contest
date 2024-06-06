import os
import io
import spotipy
import gradio as gr
import yaml
import requests, base64
from openai import OpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function, render_text_description
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.schema.agent import AgentFinish
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import AgentExecutor, load_tools
from langchain.memory import ConversationBufferMemory
from spotipy.oauth2 import SpotifyOAuth

# process config to get Nvidia and Spotify credentials
with open('agent_config.yml', 'r') as yfile:
    keys = yaml.safe_load(yfile)

os.environ["NVIDIA_API_KEY"] = keys["nvidia_api_key"]
os.environ["SPOTIPY_CLIENT_ID"] = keys["spotify_client_id"]
os.environ["SPOTIPY_CLIENT_SECRET"] = keys["spotipy_client_secret"]
os.environ["SPOTIPY_REDIRECT_URI"] = keys["spotipy_redirect_url"]


@tool
def image_analysis_tool(image_name: str) -> str:
    """vision language model that gets an image and suggests appropriate music based on its vibe + place.
    input to the tool should be a file name. """
    phi3_key=os.getenv("NVIDIA_API_KEY")
    prompt = "Suggest a search phrase to search and play music that fits the place in the image and its vibe. " \
             "Each suggested keyphrase should correspond to a genre, an artist or a track name. \
                Each suggested keyphrase should contain three words maximum. \
                Return three suggestions separated by \n. No numbering."
    invoke_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/phi-3-vision-128k-instruct"
    stream = False

    with open(image_name, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    assert len(image_b64) < 180_000, \
        "To upload larger images, use the assets API (see docs)"

    headers = {
        "Authorization": f"Bearer {phi3_key}",
        "Accept": "text/event-stream" if stream else "application/json"
        }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f'{prompt} <img src="data:image/png;base64,{image_b64}" />'
            }
        ],
        "max_tokens": 512,
        "temperature": 1.00,
        "top_p": 0.70,
        "stream": stream
    }

    result = requests.post(invoke_url, headers=headers, json=payload)
    return result.json()["choices"][0]["message"]["content"]

@tool
def search_and_play_music(searchstr: str):
    """plays music on spotify. input to the tool should be a keyphrase \
    for genre, artist or song name. this keyphrase should be short and concise."""
    scope = "user-read-playback-state,user-modify-playback-state"
    sp = spotipy.Spotify(client_credentials_manager=SpotifyOAuth(scope=scope))
    try:
        new_uri = [x["uri"] for x in sp.search(q='artist:' + searchstr, type='track')["tracks"]["items"]][0]
    except IndexError:
        new_search = searchstr.split(' ')[0]
        new_uri = [x["uri"] for x in sp.search(q='artist:' + new_search, type='track')["tracks"]["items"]][0]
    sp.start_playback(uris=[new_uri])
    return "Started playing selected track"


def vibe_agent(query: str, image_file: str):
    """
    The Agent receives the photo uploaded by the user and the query where the user
    describes their current activity. Based on these inputs, the Agent reasons about
    its course of action necessary to respond to the user request.
    The Agent has two tools: image analysis tool and Spotify tool.

    Parameters:
    query: string. User query directed to the agent.
    image_file: string. File name of the user uploaded photo.

    Returns:
    Agent final response. The Agent explains its choice of the music to play.
    """
    # define the llm
    llm = ChatNVIDIA(model="mixtral-8x7b", temperature=0.1, max_tokens=100, top_p=1.0)
    # define the tools
    custom_tools = [image_analysis_tool, search_and_play_music]
    tools = load_tools([], llm=llm)
    tools = tools + custom_tools
    # define the prompt
    system_prompt_init = f"You are very powerful assistant. the user has uploaded a photo " \
                         f"as a file named ```{image_file}```. "
    system_prompt = io.open("system_prompt.txt").read()
    prompt = ChatPromptTemplate.from_messages(
      [
        (
            "system",
            system_prompt_init+system_prompt
        ),
        ("user", "{input}+{agent_scratchpad}")
      ]
    )
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools])
    )
    # define the agent
    chat_model_with_stop = llm.bind(stop=["\nObservation"])
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | chat_model_with_stop
        | ReActJsonSingleInputOutputParser()
    )

    # instantiate AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    # get response
    response = agent_executor.invoke({"input": query})
    return response["output"]

# Run Gradio App
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Vibe Agent: plays the music that fits your current place and activity")
    image_fname = gr.Textbox(label="Image filename")
    with gr.Row():
        with gr.Column():
            query = gr.Textbox(label="Describe your activity")
            btn = gr.Button("Submit")
        with gr.Column():
            res = gr.Textbox(label="Agent Response")
    btn.click(fn=vibe_agent, inputs=[query, image_fname], outputs=[res])
gr.close_all()
demo.launch(server_name='0.0.0.0', server_port=5350)
