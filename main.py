import json
import numpy as np
import requests
from repo.LSBSteg import decode
from repo.Solvers.fox_submission_solver import generate_message_array, submit_fox_attempt
from repo.Solvers.fox_utils import split_massage_chunks
from repo.Solvers.riddle_solvers import solve_problem_solving_hard, solve_problem_solving_medium, solve_sec_hard

api_base_url = "http://13.53.169.72:5000"

team_id = 'ds42W0d'

# message = init_fox(team_id)
# print("Message:", message)

def riddle_ps_medium():
    pattern = '3[d1[e2[l]]]'
    print(solve_problem_solving_medium(pattern))


def riddle_ps_hard():
    m, n = map(int, input("m n: ").split())
    print(solve_problem_solving_hard([m, n]))


def riddle_sec_hard():
    key = "266200199BBCDFF1"
    plain_text = "0123456789ABCDEF"
    print(solve_sec_hard([key, plain_text]))


def get_remaining_attempts():
    url = f'{api_base_url}/attempts/student'
    payload = {"teamId": team_id}
    res = requests.post(url, json=payload)
    print(res.json())


def encode_decode():
    with open('game.json', 'r') as f:
        content = json.loads(f.read())
        img = content['carrier_image']
        img = np.array(img)

    enc_images = generate_message_array("Muhammad Assawalhy", img, 3)
    dec = []
    for enc in enc_images:
        dec.append(decode(enc))
    print(dec)


get_remaining_attempts()

# submit_fox_attempt(team_id)
