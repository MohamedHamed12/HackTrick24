import json
import os
from ..LSBSteg import decode
import requests
import numpy as np

api_base_url = "http://3.70.97.142:5000"
team_id = 'ds42W0d'


def get_model():
    import tensorflow as tf
    __dir__ = os.path.dirname(__file__)
    model_path = os.path.join(__dir__, "../../model1.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def get_cache_file(cache_file):
    __dir__ = os.path.dirname(__file__)
    cache_file = os.path.join(
        __dir__, ".cache", "eagle", cache_file)
    cache_dir = os.path.dirname(cache_file)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_file


def get_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            content = f.read()
            try:
                data = json.loads(content)
            except:
                data = f.read(content)
            return data

    return None


def handle_response(response: requests.Response, cache_file):
    try:
        data = response.json()
    except json.decoder.JSONDecodeError:
        data = response.text
    if response.status_code >= 400:
        raise Exception(f"{response.status_code}: {response.text}")
    with open(cache_file, "w") as f:
        f.write(response.text)
    return data


def init_eagle(team_id, use_cache):
    '''
    In this fucntion you need to hit to the endpoint to start the game as an eagle with your team id.
    If a sucessful response is returned, you will recive back the first footprints.
    '''

    cache_file = get_cache_file('game.json')

    if use_cache:
        if data := get_cache(cache_file):
            return data['footprint']

    endpoint = f"{api_base_url}/eagle/start"
    payload = {"teamId": team_id}
    response = requests.post(endpoint, json=payload)
    return handle_response(response, cache_file)['footprint']


def select_channel(model, footprint, iteration, use_cache):
    '''
    According to the footprint you recieved (one footprint per channel)
    you need to decide if you want to listen to any of the 3 channels or just skip this message.
    Your goal is to try to catch all the real messages and skip the fake and the empty ones.
    Refer to the documentation of the Footprints to know more what the footprints represent to guide you in your approach.        
    '''

    cache_file = get_cache_file(f'{iteration}/select_channel.json')

    if use_cache:
        if data := get_cache(cache_file):
            return data

    for channel in ['1', '2', '3']:
        try:
            input_data = footprint[channel]
            input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
            model.set_tensor(model.get_input_details()[
                0]['iteration'], input_data)
            model.invoke()
            output = model.get_tensor(
                model.get_output_details()[0]['iteration'])
            if output > 0.85:
                with open(cache_file, 'w') as f:
                    json.dump(channel, f)
                return channel
        except Exception as error:
            print(error)
            print("Failed to run the model, use fallback channel ", channel)
            with open(cache_file, 'w') as f:
                json.dump(channel, f)
            return channel


def skip_msg(team_id, iteration, use_cache):
    '''
    If you decide to NOT listen to ANY of the 3 channels then you need to hit the end point skipping the message.
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''

    cache_file = get_cache_file(f'{iteration}/skip_msg.json')

    if use_cache:
        if data := get_cache(cache_file):
            if data == "End of message reached":
                return None
            return json.loads(data)['nextFootprint']

    endpoint = f"{api_base_url}/eagle/skip-message"
    payload = {"teamId": team_id}
    response = requests.post(endpoint, json=payload)
    data = handle_response(response, cache_file)
    if data == "End of message reached":
        return None
    return json.loads(data)['nextFootprint']


def request_msg(team_id, channel_id, iteration, use_cache):
    '''
    If you decide to listen to any of the 3 channels then you need to hit the end point of selecting a channel to hear on (1,2 or 3)
    '''

    cache_file = get_cache_file(f'{iteration}/request_msg.json')

    if use_cache:
        if data := get_cache(cache_file):
            return data['encodedMsg']

    endpoint = f"{api_base_url}/eagle/request-message"
    payload = {"teamId": team_id, 'channelId': channel_id}
    response = requests.post(endpoint, json=payload)
    data = handle_response(response, cache_file)
    return data['encodedMsg']


def submit_msg(team_id, decoded_msg, iteration, use_cache):
    '''
    In this function you are expected to:
        1. Decode the message you requested previously
        2. call the api end point to send your decoded message  
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''

    cache_file = get_cache_file(f'{iteration}/submit_msg.json')

    if use_cache:
        if data := get_cache(cache_file):
            if data == "End of message reached":
                return None
            return json.loads(data)['nextFootprint']

    endpoint = f"{api_base_url}/eagle/submit-message"
    payload = {"teamId": team_id, 'decodedMsg': decoded_msg}
    response = requests.post(endpoint, json=payload)
    data = handle_response(response, cache_file)
    if data == "End of message reached":
        return None
    return json.loads(data)['nextFootprint']


def end_eagle(team_id, use_cache):
    '''
    Use this function to call the api end point of ending the eagle  game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    '''

    cache_file = get_cache_file('end-game.json')

    if use_cache:
        if data := get_cache(cache_file):
            return data

    endpoint = f"{api_base_url}/eagle/end-game"
    payload = {"teamId": team_id}
    response = requests.post(endpoint, json=payload)
    return handle_response(response, cache_file)


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

    use_cache = True

    model = get_model()  # keep at the top, time costly

    footprint = init_eagle(team_id, use_cache=use_cache)

    iteration = 0
    while footprint:
        iteration += 1

        channel = select_channel(model, footprint, iteration, use_cache=False)

        if channel == None:
            footprint = skip_msg(team_id, iteration, use_cache=use_cache)
        else:
            channel = int(channel)
            encoded_msg = request_msg(
                team_id, channel, iteration, use_cache=use_cache)
            decoded_msg = decode(np.array(encoded_msg))
            footprint = submit_msg(
                team_id, decoded_msg, iteration, use_cache=use_cache)

    end_eagle(team_id, use_cache=use_cache)
