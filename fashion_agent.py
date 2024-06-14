import os
import io
import yaml
import gradio as gr
from langchain.tools import tool
import requests, base64, pdb
from typing import List
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
from pydantic import BaseModel
from pydantic import Field
from openai import OpenAI


with open('agent_config.yml', 'r') as yfile:
    keys = yaml.safe_load(yfile)

os.environ["NVIDIA_API_KEY"] = keys["nvidia_api_key"]


@tool
def outfit_analysis_tool(image_name: str) -> str:
    """vision language model provides a detailed analysis of the outfit in the image
    and describes its components. input to the tool should be the name of the image to analyze."""
    api_key=os.getenv("NVIDIA_API_KEY")
    prompt = "describe the items the person is wearing in detail, including color, style, fit, silhouette."
    # call phi 3 vision instruct to analyze the look the user wants to adapt
    invoke_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/phi-3-vision-128k-instruct"
    stream = False

    with open(image_name, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    assert len(image_b64) < 180_000, \
        "To upload larger images, use the assets API (see docs)"

    headers = {
        "Authorization": f"Bearer {api_key}",
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

    response = requests.post(invoke_url, headers=headers, json=payload)
    if stream:
        text = []
        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))

    else:
        print(response.json()["choices"][0]["message"]["content"])



@tool
def image_generation_tool(prompt: str):
    """Image generation tool, generates and saves images from textual prompt.
       input to the tool is a description of the fashion look you want to generate. """
    api_key = os.getenv("NVIDIA_API_KEY")
    # call stable diffusion to generate a new fashion look
    invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-xl"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    payload = {
        "text_prompts": [
            {
                "text": "A full figure model image: "+ prompt,
                "weight": 1
            },
            {
                "text": "",
                "weight": -1
            }
        ],
        "cfg_scale": 5,
        "sampler": "K_DPM_2_ANCESTRAL",
        "seed": 0,
        "steps": 25
    }

    response = requests.post(invoke_url, headers=headers, json=payload)

    response.raise_for_status()
    response_body = response.json()
    # decode and save the generated image as .png file
    image_64_decode = base64.b64decode(response_body["artifacts"][0]['base64'])
    image_result = open('gen_result.png', 'wb')
    image_result.write(image_64_decode)
    return "Image saved"



def fashion_agent(query: str, image_file: str):
    """
        The Agent receives the photo uploaded by the user and the query where the user
        asks how to adapt the outfit in the photo to some new purpose.
        Example: how can I wear this outfit to a party?
        Based on these inputs, the Agent reasons about
        its course of action necessary to respond to the user request.
        The Agent has two tools: image analysis tool and image generation tool.

        Parameters:
        query: string. User query directed to the agent.
        image_file: string. File name of the user uploaded photo.

        Returns:
        Agent final response. The Agent explains its modifications to the original look.
        """
    # define the llm
    llm = ChatNVIDIA(model="ai-mixtral-8x7b-instruct", temperature=0.1, max_tokens=100, top_p=1.0)
    # set up tools
    custom_tools = [outfit_analysis_tool, image_generation_tool]
    tools = load_tools([], llm=llm)
    tools = tools + custom_tools
    # define the prompt
    system = f"You are very powerful fashion assistant. the user has uploaded a photo \
        as a file with the name {image_file}."
    system_prompt = io.open("fashion_prompt.txt").read()
    prompt = ChatPromptTemplate.from_messages(
      [
        (
            "system",
            system+system_prompt
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
    response = agent_executor.invoke({"input": query})
    return response["output"]

# Run Gradio App
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Fashion Agent: helps you repurpose your outfits")
    image_fname = gr.Textbox(label="Image filename")
    with gr.Row():
        with gr.Column():
            query = gr.Textbox(label="What do you want to do with this outfit?")
            btn = gr.Button("Submit")
        with gr.Column():
            res = gr.Textbox(label="Agent Response")
    btn.click(fn=fashion_agent, inputs=[query, image_fname], outputs=[res])
gr.close_all()
demo.launch(server_name='0.0.0.0', server_port=5450)
