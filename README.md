# Cool LLM Agents
Contest information here https://www.nvidia.com/en-us/ai-data-science/generative-ai/developer-contest-with-langchain/

## Vibe Agent
Want to hear some music that fits the vibe of the place you are now and your current activity ? Just upload the photo and describe your activity. The Agent will choose some appropriate music for you and play it. I am pretty sure you will appreciate its choices.

## Fashion Agent
Sometimes you see a cute outfit, but you want it to be adapted to some other circumstances. For instance, you want to give it some more festive look. Or, on the contrary, a more downplay look. The Fashion Agent will help. Upload your image and describe how you want to adapt it. The Agent will propose adaptations and generate an image of the updated look for you. 

## Running the agents
- Install requirements (for python 3.9)

```shell
    pip install -r requirements.txt      
```
- Get NVIDIA API key
  Register here https://build.nvidia.com/explore/discover and click on the model of your choice. You should see the button "Get API Key" in code examples. Add the API key to the agent_config.yml file. 

- Get Spotify access credentials
  Follow the instructions here https://developer.spotify.com/documentation/web-api. Add the credentials to the agent_config.yml file.  It is likely that to be able to use the Spotify player in this agent you will need a Premium account. 

- To run the Vibe agent:

```shell
      python vibe_agent.py 
```
- To run the Fashion agent:

```shell
      python fashion_agent.py 
```

Place your photos in 'data' folder. In the app, indicate the image file name together with the folder, e.g. 'data/my_image.jpg'.  
