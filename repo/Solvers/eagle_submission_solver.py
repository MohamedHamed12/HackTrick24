import json
import os
from ..LSBSteg import decode
import requests
import numpy as np
import tensorflow as tf

api_base_url = "http://3.70.97.142:5000"
team_id = 'ds42W0d'
__dir__ = os.path.dirname(__file__)
model_path = os.path.join(__dir__, "../../model1.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()


def get_cache(cache_file):
    __dir__ = os.path.dirname(__file__)
    cache_dir = os.path.join(__dir__, ".cache", "eagle")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    file = os.path.join(cache_dir, cache_file)
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.loads(f.read())
            return data

    return None


def handle_response(response: requests.Response, cache_file, is_text=False):
    if response.status_code == 200 or response.status_code == 201:
        data = response.text if is_text else response.json()
        with open(get_cache(cache_file), "w") as f:
            f.write(data.text)
        return data
    else:
        raise f"{response.status_code}: {response.text}"


def init_eagle(team_id, use_cache):
    '''
    In this fucntion you need to hit to the endpoint to start the game as an eagle with your team id.
    If a sucessful response is returned, you will recive back the first footprints.
    '''

    cache_file = 'game.json'

    if use_cache:
        if data := get_cache(cache_file):
            return data

    endpoint = f"{api_base_url}/eagle/start"
    payload = {"teamId": team_id}
    response = requests.post(endpoint, json=payload)
    return handle_response(response, cache_file)


def select_channel(footprint):
    '''
    According to the footprint you recieved (one footprint per channel)
    you need to decide if you want to listen to any of the 3 channels or just skip this message.
    Your goal is to try to catch all the real messages and skip the fake and the empty ones.
    Refer to the documentation of the Footprints to know more what the footprints represent to guide you in your approach.        
    '''

    for channel in ['1', '2', '3']:
        input_data = footprint[channel]
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        interpreter.set_tensor(interpreter.get_input_details()[
            0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(
            interpreter.get_output_details()[0]['index'])
        if output > 0.85:
            return channel


def skip_msg(team_id, index, use_cache):
    '''
    If you decide to NOT listen to ANY of the 3 channels then you need to hit the end point skipping the message.
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''

    cache_file = 'skip_{index}.json'

    if use_cache:
        if data := get_cache(cache_file):
            return data

    endpoint = f"{api_base_url}/eagle/skip-message"
    payload = {"teamId": team_id}
    response = requests.post(endpoint, json=payload)
    data = handle_response(response, cache_file, True)
    if data == "End of message reached":
        return None
    return json.loads(data)['nextFootprint']


def request_msg(team_id, channel_id, index, use_cache):
    '''
    If you decide to listen to any of the 3 channels then you need to hit the end point of selecting a channel to hear on (1,2 or 3)
    '''
    cache_file = f'request_msg_{index}.json'

    if use_cache:
        if data := get_cache(cache_file):
            return data

    endpoint = f"{api_base_url}/eagle/request-message"
    payload = {"teamId": team_id, 'channelId': channel_id}
    response = requests.post(endpoint, json=payload)
    data = handle_response(response, cache_file)['encodedMsg']
    return data


def submit_msg(team_id, decoded_msg, index, use_cache):
    '''
    In this function you are expected to:
        1. Decode the message you requested previously
        2. call the api end point to send your decoded message  
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''

    cache_file = f'submit_{index}.json'

    if use_cache:
        if data := get_cache(cache_file):
            return data

    endpoint = f"{api_base_url}/eagle/submit-message"
    payload = {"teamId": team_id, 'decodedMsg': decoded_msg}
    response = requests.post(endpoint, json=payload)
    data = handle_response(response, cache_file, True)
    if data == "End of message reached":
        return None
    return json.loads(data)['nextFootprint']


def end_eagle(team_id, use_cache):
    '''
    Use this function to call the api end point of ending the eagle  game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    '''

    cache_file = 'end-game.json'

    if use_cache:
        if data := get_cache(cache_file):
            return data

    endpoint = f"{api_base_url}/eagle/end-game"
    payload = {"teamId": team_id}
    response = requests.post(endpoint, json=payload)
    handle_response(response, cache_file, True)


def submit_eagle_attempt(team_id):
    '''
     Call this function to start playing as an eagle. 
     You should submit with your own team id that was sent to you in the email.
     Remeber you have up to 15 Submissions as an Eagle In phase1.
     In this function you should:
        1. Initialize the game as fox 
        2. Solve the footprints to know which channel to listen on if any.
        3. Select a channel to hear on OR send skip request.
        4. Submit your answer in case you listened on any channel
        5. End the Game
    '''

    footprint = init_eagle(team_id, use_cache=False)['footprint']
    print(footprint)

    index = 0
    while footprint:
        index += 1

        channel = select_channel(footprint)

        if channel == None:
            footprint = skip_msg()
        else:
            encoded_msg = request_msg(team_id, channel, index, use_cache=False)
            decoded_msg = decode(encoded_msg)
            footprint = submit_msg(
                team_id, channel, decoded_msg, index, use_cache=False)

    end_eagle()
