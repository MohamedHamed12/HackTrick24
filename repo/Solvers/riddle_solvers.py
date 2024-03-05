import pandas as pd
from pyparsing import deque
import numpy as np
import torch
from SteganoGAN.utils import *
from sklearn.ensemble import RandomForestRegressor
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import spacy
import inflect
import cv2

def solve_cv_easy(test_case: tuple) -> list:
    shredded_image, shred_width = test_case
    shredded_image = np.array(shredded_image)
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing a shredded image.
        - An integer representing the shred width in pixels.

    Returns:
    list: A list of integers representing the order of shreds. When combined in this order, it builds the whole image.
    """
    return []


def solve_cv_medium(input: tuple) -> list:
    combined_image_array, patch_image_array = input
    combined_image = np.array(combined_image_array, dtype=np.uint8)
    patch_image = np.array(patch_image_array, dtype=np.uint8)

    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing the RGB base image.
        - A numpy array representing the RGB patch image.

    Returns:
    list: A list representing the real image.
    """

    def compute_homography_matrix(src_points, dest_points):
        if len(src_points) < 4 or len(dest_points) < 4:
            raise ValueError("At least 4 corresponding points are required.")

        A = []
        for i in range(len(src_points)):
            x, y = src_points[i]
            u, v = dest_points[i]
            A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
            A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])

        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1, :].reshape(3, 3)
        H /= H[2, 2]

        return H

    def getCorrespondences(img1, img2, max_matches=50, ratio_threshold=0.75, ransac_reproj_threshold=5.0,
                           max_iterations=1000):
        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

        good_matches = sorted(good_matches, key=lambda x: x.distance)
        if len(good_matches) > max_matches:
            good_matches = good_matches[:max_matches]

        src_points = np.float32(
            [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dest_points = np.float32(
            [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        best_inlier_count = 0
        best_inliers = []

        for _ in range(max_iterations):
            random_indices = np.random.choice(
                len(good_matches), 4, replace=False)
            sampled_src = np.float32(
                [kp1[good_matches[i].queryIdx].pt for i in random_indices])
            sampled_dest = np.float32(
                [kp2[good_matches[i].trainIdx].pt for i in random_indices])

            H = compute_homography_matrix(sampled_src, sampled_dest)

            transformed_src = np.dot(H, np.vstack(
                (src_points.T, np.ones(len(src_points)))))

            # Normalize the transformed points
            transformed_src = transformed_src[:2, :] / transformed_src[2, :]
            transformed_src = transformed_src.T

            # Compute the reprojection error
            errors = np.linalg.norm(transformed_src - dest_points, axis=1)

            # Count inliers based on the reprojection error threshold
            inliers = np.where(errors < ransac_reproj_threshold)[0]
            inlier_count = len(inliers)

            # Update the best set of inliers
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers

        inlier_matches = [good_matches[i] for i in best_inliers]
        src_points = np.float32(
            [kp1[m.queryIdx].pt for m in inlier_matches]).reshape(-1, 2)
        dest_points = np.float32(
            [kp2[m.trainIdx].pt for m in inlier_matches]).reshape(-1, 2)

        return src_points, dest_points, kp1, kp2, inlier_matches

    def map_points_with_homography(src_points, homography_matrix):
        homogenous_src_points = np.column_stack(
            (src_points, np.ones((len(src_points), 1))))
        dest_points_homogeneous = np.dot(
            homogenous_src_points, homography_matrix.T)
        dest_points = dest_points_homogeneous[:,
                                              :2] / dest_points_homogeneous[:, 2:]
        return dest_points

    def remove_patch(base_image, patch_coordinates):
        patched_removed_image = base_image.copy()
        mask = np.zeros_like(base_image, dtype=np.uint8)
        top_left, bottom_right = patch_coordinates
        mask[round(top_left[1]):round(bottom_right[1]), round(
            top_left[0]):round(bottom_right[0])] = 255
        patched_removed_image = cv2.inpaint(
            patched_removed_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return patched_removed_image

    def interpolate(base_image):
        kernel = np.ones((5, 5), np.uint8)
        interpolated_image = cv2.inpaint(
            base_image, cv2.bitwise_not(base_image), 5, cv2.INPAINT_TELEA)
        return interpolated_image

    base_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
    patch_image = cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR)

    src_points, dest_points, kp1, kp2, good_matches = getCorrespondences(
        patch_image, base_image)

    patch_height, patch_width = patch_image.shape
    homography_matrix = compute_homography_matrix(src_points, dest_points)
    book_corners = np.float32(
        [[0, 0], [patch_width, patch_height]]).reshape(-1, 2)
    patch_coordinates = map_points_with_homography(
        book_corners, homography_matrix)

    base_image_patched_removed = remove_patch(base_image, patch_coordinates)
    result_image = interpolate(base_image_patched_removed)
    # Convert the CV2 image to a NumPy array
    image_np = np.array(result_image)
    # Convert the NumPy array to a list
    image_list = image_np.tolist()
    return image_list


def solve_cv_hard(input: tuple) -> int:
    # extracted_question, image = test_case
    # image = np.array(image)
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A string representing a question about an image.
        - An RGB image object loaded using the Pillow library.

    Returns:
    int: An integer representing the answer to the question about the image.
    """

    # Load English tokenizer, tagger, parser, NER, and word vectors
    nlp = spacy.load("en_core_web_sm")
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    coco_class_to_id = {class_name: i + 1 for i,
                        class_name in enumerate(coco_classes)}
    p = inflect.engine()

    def count_objects(input):
        question, image = input

        # Define transformations to preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
        ])

        # Preprocess the image
        image = preprocess(image).unsqueeze(0)

        # Load pre-trained Faster R-CNN model
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()

        # Perform object detection
        with torch.no_grad():
            predictions = model(image)

        # Extract object of interest from the question
        doc = nlp(question)
        object_of_interest = None
        for token in doc:
            if token.pos_ == 'NOUN':
                object_of_interest = token.text.lower()
                break

        object_of_interest = p.singular_noun(
            object_of_interest) or object_of_interest
        relevant_labels = [coco_class_to_id.get(
            object_of_interest.lower(), None) + 1]

        labels = predictions[0]['labels']
        scores = predictions[0]['scores']

        # Filter out irrelevant objects based on labels, confidence scores, and object of interest
        min_score_threshold = 0.5  # Example: minimum confidence score threshold
        relevant_count = sum(
            1 for label, score in zip(labels, scores) if label in relevant_labels and score >= min_score_threshold)

        return relevant_count

    return count_objects(input)


def solve_ml_easy(data: pd.DataFrame) -> list:
    """
    This function takes a pandas DataFrame as input and returns a list as output.

    Parameters:
    input (pd.DataFrame): A pandas DataFrame representing the input data.

    Returns:
    list: A list of floats representing the output of the function.
    """

    # Convert 'date' column to datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Feature Engineering
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['day_of_year'] = data['timestamp'].dt.dayofyear

    # Split data into training and testing sets
    X = data[['day_of_week', 'day_of_year']]
    y = data['visits']

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Forecasting for the next 50 days
    future_dates = pd.date_range(
        start=data['timestamp'].iloc[-1], periods=50, freq='D')
    future_features = pd.DataFrame({
        'day_of_week': future_dates.dayofweek,
        'day_of_year': future_dates.dayofyear
    })
    forecast = model.predict(future_features)
    return forecast.tolist()


# def solve_sec_medium(input: torch.Tensor) -> str:
#     img = torch.tensor(img)
#     """
#     This function takes a torch.Tensor as input and returns a string as output.

#     Parameters:
#     input (torch.Tensor): A torch.Tensor representing the image that has the encoded message.

#     Returns:
#     str: A string representing the decoded message from the image.
#     """
#     return ''

def solve_sec_hard(input: tuple) -> str:
    """
    This function takes a tuple as input and returns a list a string.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A key 
        - A Plain text.

    Returns:
    list:A string of ciphered text
    """

    def hex2bin(s):
        mp = {'0': "0000",
              '1': "0001",
              '2': "0010",
              '3': "0011",
              '4': "0100",
              '5': "0101",
              '6': "0110",
              '7': "0111",
              '8': "1000",
              '9': "1001",
              'A': "1010",
              'B': "1011",
              'C': "1100",
              'D': "1101",
              'E': "1110",
              'F': "1111"}
        bin = ""
        for i in range(len(s)):
            bin = bin + mp[s[i]]
        return bin

    # Binary to hexadecimal conversion

    def bin2hex(s):
        mp = {"0000": '0',
              "0001": '1',
              "0010": '2',
              "0011": '3',
              "0100": '4',
              "0101": '5',
              "0110": '6',
              "0111": '7',
              "1000": '8',
              "1001": '9',
              "1010": 'A',
              "1011": 'B',
              "1100": 'C',
              "1101": 'D',
              "1110": 'E',
              "1111": 'F'}

        hex = ""

        for i in range(0, len(s), 4):
            ch = ""
            ch = ch + s[i]
            ch = ch + s[i + 1]
            ch = ch + s[i + 2]
            ch = ch + s[i + 3]
            hex = hex + mp[ch]

        return hex

    def bin2dec(binary):
        binary1 = binary
        decimal, i, n = 0, 0, 0
        while (binary != 0):
            dec = binary % 10
            decimal = decimal + dec * pow(2, i)
            binary = binary//10
            i += 1
        return decimal

    def dec2bin(num):
        res = bin(num).replace("0b", "")
        if (len(res) % 4 != 0):
            div = len(res) / 4
            div = int(div)
            counter = (4 * (div + 1)) - len(res)
            for i in range(0, counter):
                res = '0' + res
        return res

    def permute(k, arr, n):
        permutation = ""
        for i in range(0, n):
            permutation = permutation + k[arr[i] - 1]
        return permutation

    def shift_left(k, nth_shifts):
        s = ""
        for i in range(nth_shifts):
            for j in range(1, len(k)):
                s = s + k[j]
            s = s + k[0]
            k = s
            s = ""
        return k

    def xor(a, b):
        ans = ""
        for i in range(len(a)):
            if a[i] == b[i]:
                ans = ans + "0"
            else:
                ans = ans + "1"
        return ans

    initial_perm = [58, 50, 42, 34, 26, 18, 10, 2,
                    60, 52, 44, 36, 28, 20, 12, 4,
                    62, 54, 46, 38, 30, 22, 14, 6,
                    64, 56, 48, 40, 32, 24, 16, 8,
                    57, 49, 41, 33, 25, 17, 9, 1,
                    59, 51, 43, 35, 27, 19, 11, 3,
                    61, 53, 45, 37, 29, 21, 13, 5,
                    63, 55, 47, 39, 31, 23, 15, 7]

    exp_d = [32, 1, 2, 3, 4, 5, 4, 5,
             6, 7, 8, 9, 8, 9, 10, 11,
             12, 13, 12, 13, 14, 15, 16, 17,
             16, 17, 18, 19, 20, 21, 20, 21,
             22, 23, 24, 25, 24, 25, 26, 27,
             28, 29, 28, 29, 30, 31, 32, 1]

    per = [16, 7, 20, 21,
           29, 12, 28, 17,
           1, 15, 23, 26,
           5, 18, 31, 10,
           2, 8, 24, 14,
           32, 27, 3, 9,
           19, 13, 30, 6,
           22, 11, 4, 25]

    sbox = [[[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
            [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
            [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
            [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],

            [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
            [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
            [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
            [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],

            [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
            [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
            [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
            [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],

            [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
            [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
            [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
            [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],

            [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
            [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
            [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
            [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],

            [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
            [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
            [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
            [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],

            [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
            [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
            [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
            [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],

            [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
            [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
            [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
            [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]]]

    final_perm = [40, 8, 48, 16, 56, 24, 64, 32,
                  39, 7, 47, 15, 55, 23, 63, 31,
                  38, 6, 46, 14, 54, 22, 62, 30,
                  37, 5, 45, 13, 53, 21, 61, 29,
                  36, 4, 44, 12, 52, 20, 60, 28,
                  35, 3, 43, 11, 51, 19, 59, 27,
                  34, 2, 42, 10, 50, 18, 58, 26,
                  33, 1, 41, 9, 49, 17, 57, 25]

    def encrypt(pt, rkb, rk):
        pt = hex2bin(pt)

        pt = permute(pt, initial_perm, 64)

        left = pt[0:32]
        right = pt[32:64]
        for i in range(0, 16):
            # Expansion D-box: Expanding the 32 bits data into 48 bits
            right_expanded = permute(right, exp_d, 48)

            # XOR RoundKey[i] and right_expanded
            xor_x = xor(right_expanded, rkb[i])

            # S-boxex: substituting the value from s-box table by calculating row and column
            sbox_str = ""
            for j in range(0, 8):
                row = bin2dec(int(xor_x[j * 6] + xor_x[j * 6 + 5]))
                col = bin2dec(
                    int(xor_x[j * 6 + 1] + xor_x[j * 6 + 2] + xor_x[j * 6 + 3] + xor_x[j * 6 + 4]))
                val = sbox[j][row][col]
                sbox_str = sbox_str + dec2bin(val)

            sbox_str = permute(sbox_str, per, 32)

            result = xor(left, sbox_str)
            left = result

            if (i != 15):
                left, right = right, left

        combine = left + right

        cipher_text = permute(combine, final_perm, 64)
        return cipher_text

    key, pt = input

    key = hex2bin(key)

    keyp = [57, 49, 41, 33, 25, 17, 9,
            1, 58, 50, 42, 34, 26, 18,
            10, 2, 59, 51, 43, 35, 27,
            19, 11, 3, 60, 52, 44, 36,
            63, 55, 47, 39, 31, 23, 15,
            7, 62, 54, 46, 38, 30, 22,
            14, 6, 61, 53, 45, 37, 29,
            21, 13, 5, 28, 20, 12, 4]

    key = permute(key, keyp, 56)

    shift_table = [1, 1, 2, 2,
                   2, 2, 2, 2,
                   1, 2, 2, 2,
                   2, 2, 2, 1]

    key_comp = [14, 17, 11, 24, 1, 5,
                3, 28, 15, 6, 21, 10,
                23, 19, 12, 4, 26, 8,
                16, 7, 27, 20, 13, 2,
                41, 52, 31, 37, 47, 55,
                30, 40, 51, 45, 33, 48,
                44, 49, 39, 56, 34, 53,
                46, 42, 50, 36, 29, 32]

    # Splitting
    left = key[0:28]  # rkb for RoundKeys in binary
    right = key[28:56]  # rk for RoundKeys in hexadecimal

    rkb = []
    rk = []
    for i in range(0, 16):
        left = shift_left(left, shift_table[i])
        right = shift_left(right, shift_table[i])

        combine_str = left + right

        # Compression of key from 56 to 48 bits
        round_key = permute(combine_str, key_comp, 48)

        rkb.append(round_key)
        rk.append(bin2hex(round_key))

    cipher_text = bin2hex(encrypt(pt, rkb, rk))
    return cipher_text


def solve_problem_solving_easy(input: tuple) -> list:
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A list of strings representing a question.
        - An integer representing a key.

    Returns:
    list: A list of strings representing the solution to the problem.
    """
    words, X = input
    word_counts = {}
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    sorted_words = sorted(word_counts.keys(),
                          key=lambda word: (-word_counts[word], word))
    return sorted_words[:X]


def solve_problem_solving_medium(input: str) -> str:
    """
    This function takes an encoded string as input and returns the decoded string.

    Parameters:
    input (str): An encoded string.

    Returns:
    str: The decoded string.
    """

    def is_int(ch):
        return ch in '0123456789'

    stack = deque()
    current_string = ""

    tokens = []
    for c in input:
        if len(tokens) and is_int(tokens[-1][0]) and is_int(c):
            tokens[-1] += c
        else:
            tokens.append(c)

    for t in tokens:
        if t == '[':
            pass
        elif t == ']':
            rep, st = stack.pop()
            current_string += current_string[st:] * (rep - 1)
        elif is_int(t[0]):
            stack.append((int(t), len(current_string)))
        else:
            current_string += t

    return current_string


def solve_problem_solving_hard(input: tuple) -> int:
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two integers representing m and n.

    Returns:
    int: An integer representing the solution to the problem.
    """
    m, n = input
    dp = [[0]*(n + 1) for i in range(m + 1)]
    dp[0][1] = 1
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m][n]


riddle_solvers = {
    'cv_easy': solve_cv_easy,
    'cv_medium': solve_cv_medium,
    'cv_hard': solve_cv_hard,
    'ml_easy': solve_ml_easy,
    'ml_medium': solve_ml_medium,
    # 'sec_medium_stegano': solve_sec_medium,
    'sec_hard': solve_sec_hard,
    'problem_solving_easy': solve_problem_solving_easy,
    'problem_solving_medium': solve_problem_solving_medium,
    'problem_solving_hard': solve_problem_solving_hard
}
