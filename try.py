import os
import json
import base64
import boto3
import io
import awsiot.greengrasscoreipc
from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2
from awsiot.greengrasscoreipc.client import SubscribeToTopicStreamHandler
from awsiot.greengrasscoreipc.model import SubscribeToTopicRequest
import numpy as np
from PIL import Image
import torch
import time
from io import BytesIO
from facenet_pytorch import MTCNN

# ---------- Settings ----------
TOPIC = "clients/1233521057-IoTThing"
SQS_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/686255976854/1233521057-req-queue" 
RESP_QUEUE= "https://sqs.us-east-1.amazonaws.com/686255976854/1233521057-resp-queue"

# ---------- AWS Clients ----------
sqs_client = boto3.client('sqs', region_name="us-east-1")
ipc_client = GreengrassCoreIPCClientV2()

# ---------- Face Detection Class ----------
class FaceDetection:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)  # MTCNN init

    def image_to_base64_string(self, image):
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

    def base64_to_image(self, base64_string):
        base64_string = base64_string.split(",")[-1]  # handles with/without data:image prefix
        return Image.open(BytesIO(base64.b64decode(base64_string)))
    
    def face_detection_func(self, image):
        img     = image.convert("RGB")
        img     = np.array(img)
        img     = Image.fromarray(img)
        face, prob = self.mtcnn(img, return_prob=True, save_path=None)

        if face != None:

            face_img = face - face.min() 
            face_img = face_img / face_img.max() 
            face_img = (face_img * 255).byte().permute(1, 2, 0).numpy() 

            face_pil        = Image.fromarray(face_img, mode="RGB")
            return face_pil
        else:
            print(f"No face is detected")
            return None

fd = FaceDetection()

def on_stream_event(event):
    message = str(event.binary_message.message, 'utf-8')
    # topic = event.binary_message.context.topic
    # print('Received new message on topic %s: %s' % (topic, message))
    handle_incoming_message(message)

# ---------- Message Handler ----------
def handle_incoming_message(message):
    print("Received MQTT message...")
    payload = json.loads(message)
    print(payload)
    base64_data = payload.get('encoded')
    request_id = payload.get('request_id')
    file_name = payload.get('filename')

    if not base64_data or not request_id or not file_name:
        print("Invalid payload received, missing required fields.")
        return

    try:
        img = fd.base64_to_image(base64_data)
        cropped_img = fd.face_detection_func(img)
        face_b64 = fd.image_to_base64_string(cropped_img)
        if face_b64:
            new_message = {
            "request_id": request_id,
            "filename": file_name,
            "content": face_b64
            }
            # Send to SQS
            sqs_client.send_message(
                QueueUrl=SQS_QUEUE_URL,
                MessageBody=json.dumps(new_message)
            )
            #print(f"Face detected and message sent to SQS for request_id={request_id}")
        else:
            print("No face detected. Nothing sent to SQS.")

    except Exception as e:
        print(f"Error handling MQTT message: {e}")

# ---------- Main ----------
if __name__ == '__main__':
    ipc_client.subscribe_to_topic(topic=TOPIC, on_stream_event=on_stream_event)
    print(f"Subscribed to topic {TOPIC}")
    while True:
        time.sleep(0.01)




