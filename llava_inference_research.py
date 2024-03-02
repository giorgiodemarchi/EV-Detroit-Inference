from PIL import Image
import io
import json
import base64
import requests
import re
import cv2 

def read_images(path):
    images = [Image.open(path + f"image_{i}.png") for i in [0,2,4]]
    return images

def resize_and_encode_image(image, output_size=(300, 300)):
    image.thumbnail(output_size)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='png')
    image_bytes = img_byte_arr.getvalue()
    encoded_string = base64.b64encode(image_bytes).decode('utf-8')
    return encoded_string


def predict(folder_path, controller_url = "http://0.0.0.0:10000/worker_generate_stream"):

    frames = read_images(folder_path)
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
    prompt += f"In order to install a curbside EV charging station such as the one in this picture <image>."
    prompt += f"there must be a parking spot, enough space on the sidewalk, and there must not be any impediment such as fire hydrants."
    prompt += f"Consider the sidewalk in the pictures. Would it be feasible to install a curbside EV charging station at this exact location? </s> ASSISTANT:" 

    
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


if __name__ == "__main__":
    data_path = 'example_data/47_225_42.32507295974896_-83.0523860567444/'
    controller_url = "http://0.0.0.0:10000/worker_generate_stream"

    answer = predict(data_path, controller_url = controller_url)
    print(answer)