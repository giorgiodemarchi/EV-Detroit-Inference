from PIL import Image
import io
import json
import base64
import requests
import re
from openai import OpenAI
import boto3
import numpy as np
import pandas as pd

AWS_ACCESS_KEY_ID = "ahah"
AWS_SECRET_ACCESS_KEY_ID = "ahahah"
OPENAI_API_KEY = "AHAH"
BUCKET_NAME = 'detroit-project-data-bucket'
DIRECTORY_NAME = 'DetroitImageDataset_v2/'

s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY_ID)

def get_folder_names():
    paginator = s3_client.get_paginator('list_objects_v2')
    response_iterator = paginator.paginate(
        Bucket=BUCKET_NAME,
        Prefix=DIRECTORY_NAME,
        Delimiter='/'
    )

    folder_names = []
    for response in response_iterator:
        if response.get('CommonPrefixes') is not None:
            for prefix in response.get('CommonPrefixes'):
                # Extract the folder name from the prefix key
                folder_name = prefix.get('Prefix')
                # Removing the base directory and the trailing slash to get the folder name
                folder_name = folder_name[len(DIRECTORY_NAME):].strip('/')
                folder_names.append(folder_name)

    return folder_names

def read_images(path):
    for i in [0,2,4]:
        image_path = f"{DIRECTORY_NAME}/{path}/image_{i}.png"
        image_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=image_path)
        image = Image.open(io.BytesIO(image_obj['Body'].read()))
        yield image

def resize_and_encode_image(image, output_size=(300, 300)):
    image.thumbnail(output_size)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='png')
    image_bytes = img_byte_arr.getvalue()
    encoded_string = base64.b64encode(image_bytes).decode('utf-8')
    return encoded_string


def predict(folder_path, controller_url = "http://0.0.0.0:10000/worker_generate_stream"):

    frames = list(read_images(folder_path))
    encoded_images = [resize_and_encode_image(image) for image in frames]
    number_of_images = len(encoded_images)
    image_token_string = ""
    for _ in range(number_of_images):
        image_token_string += "<image> "

    # Example image
    example_image = Image.open('example_data/example_ev_station.jpg')
    example_image = resize_and_encode_image(example_image)
    encoded_images.append(example_image)

    prompt = f"USER: Here are {number_of_images} images of a sidewalk location in the city of Detroit: {image_token_string}."
    prompt += f"In order to install a curbside EV charging station such as the one in this picture <image>,"
    prompt += f"there must be a parking spot on the side of the street, enough space on the sidewalk, and there must not be any impediment such as fire hydrants, trees or tram rails."
    prompt += f"Consider the sidewalk in the pictures. Would it be feasible to install a curbside EV charging station at this exact location? Answer yes or no. </s> ASSISTANT:" 
    
    data = {
        "prompt": prompt,
        "images": encoded_images,
        "stop": "</s>",
        "model": "llava-v1.5-13b",
    }

    headers = {"User-Agent": "LLaVA Client"}

    response = requests.post(controller_url, json=data, headers = headers, stream=True)

    accumulated_response = []
    try:
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:
                # Directly append chunk without manipulation
                decoded_chunk = chunk.decode('utf-8').rstrip('\0')
                accumulated_response.append(decoded_chunk)
        # Join accumulated chunks and then process as a whole
        input_string = decoded_chunk.replace('\0','')
        matches = re.findall(r'\{(.*?)\}', input_string)

        # The last match is the content of the last set of curly brackets
        last_match = matches[-1].strip() if matches else None
        final_answer = last_match.split("ASSISTANT:")[1].split("\", \"error_code\"")[0]

        return final_answer

    except requests.exceptions.ChunkedEncodingError as e:
        print("Error Reading Stream:", e)
    except json.JSONDecodeError as e:
        print("Error Decoding JSON:", e)

def process_with_openai(answer):
    client = OpenAI(api_key = OPENAI_API_KEY)
    response = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    response_format={ "type": "json_object" },
    messages=[
        {"role": "system", "content": "You are a helpful assistant designed to output JSON. You will interpret an answer text provided by an expert and classify as 1 or 0. The returned json will include an attribute 'feasible' that must be associated with either one of those labels."},
        {"role": "user", "content": f"Is this response indicative that it is feasible or infeasible to install a curbside EV charging station? {answer}"}
    ])
    return response.choices[0].message.content

if __name__ == "__main__":
    
    ids = []
    angles = []
    latitudes = []
    longitudes = []
    llava_answers = []
    labels = []

    controller_url = "http://0.0.0.0:10000/worker_generate_stream"

    folders = get_folder_names()

    i=0
    for datapoint in folders:
        point_id, angle, latitude, longitude = datapoint.split('_')
        ids.append(point_id)
        angles.append(angle)
        latitudes.append(latitude)
        longitudes.append(longitude)

        answer = predict(datapoint, controller_url = controller_url) 
        llava_answers.append(answer)
        print("LLaVA ANSWER: ")
        print(answer)
        
        json_str = process_with_openai(answer)
        dict_data = json.loads(json_str)
        labels.append(dict_data["feasible"])
        print("JSON:")
        print(json_str)

        i+=1
        if i%10==0:
            print(f"Processed {i} datapoint ({np.round(i/len(folders), 2)*100} %)")

    
    data = {
        "id": ids,
        "angle": angles,
        "latitude": latitudes,
        "longitude": longitudes,
        "predicted_label": labels,
        "llava_response": llava_answers
    }

    df = pd.DataFrame(data)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=BUCKET_NAME, Body=csv_buffer.getvalue(), Key='llava_predictions.csv')