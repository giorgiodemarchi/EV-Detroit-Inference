from PIL import Image
import io
import json
import base64
import requests
import re
import cv2 


def read_video(video_path, freq):
    """
    Reads video and capture a frame every 5 seconds.
    video_path: can also be a URL
    freq: frequency of extracted frames. E.g. if freq = 5, one frame every 5 seconds is extracted
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # number of frames to skip
    skip_frames = int(fps * freq)

    # Extract a frame every 3 secs
    frame_count = 0
    extracted_frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break  # Exit the loop if we've reached the end of the video

        # Check if this frame is at a 5-second interval
        if frame_count % skip_frames == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            extracted_frames.append(pil_img)

        frame_count += 1

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
    return extracted_frames

def resize_and_encode_image(image, output_size=(300, 300)):
    image.thumbnail(output_size)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='png')
    image_bytes = img_byte_arr.getvalue()
    encoded_string = base64.b64encode(image_bytes).decode('utf-8')
    return encoded_string


def predict(video_file, mode, controller_url = "http://0.0.0.0:10000/worker_generate_stream", freq = 5):
    """
    video_file: either a local file or a url
    mode: "SFX" / "AMBIENCE" / "CAPTION"
    controller_url: 
    freq: frequency of frames extraction from video.

    TODO: Scene splitting and logic to provide multiple scenes
    """
    frames = read_video(video_file, freq)

    encoded_images = [resize_and_encode_image(image) for image in frames]
    number_of_images = len(encoded_images)


    image_token_string = ""
    for _ in range(number_of_images):
        image_token_string += "<image> "

    # prompt = f"USER: Here are {number_of_images} frames extracted from a video, in chronological order. Describe what is happening in the video. {image_token_string} </s>ASSISTANT:"
    prompt = f"USER: Here are {number_of_images} frames extracted from a video, in chronological order: {image_token_string}."
    if mode == 'CAPTION':
        prompt += "Describe what is happening in the video. </s> ASSISTANT:"
    else:
        prompt += "I want you to assist me in desining the best sound for this video. Using your understanding of what is happening in the video, including actions, objects, and scene, answer the following question: "
        if mode == 'SFX':
            prompt += "What sound effects would you expect to hear when watching the video? Provide a description suitable for searching in a sound library. </s> ASSISTANT:"
        else:
            prompt += "What ambience sounds would you expect to hear when watching the video? Provide a description suitable for searching in a sound library. </s> ASSISTANT:"

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
        # complete_response = ''.join(accumulated_response).replace('\0', '')  # Removing null characters globally
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
    video_file = 'cyclist.mp4'  # Can also be a url
    controller_url = "http://0.0.0.0:10000/worker_generate_stream"
    freq = 5
    mode = "CAPTION"

    answer = predict(video_file, mode, controller_url = controller_url, freq = freq)
    print(answer)