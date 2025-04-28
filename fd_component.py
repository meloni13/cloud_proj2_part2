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
TOPIC = f"clients/1233521057-IoTThing"
SQS_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/686255976854/1233521057-req-queue" 

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
        encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_string = f"data:image/jpeg;base64,{encoded_string}"
        return base64_string

    def base64_to_image(self, base64_string):
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",")[1]
        image_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(image_data))
        return img
    
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

# ---------- Greengrass MQTT Handler ----------
def subscribe_to_mqtt():
    class StreamHandler(SubscribeToTopicStreamHandler):
        def on_stream_event(self, event):
            message = event.payload.decode('utf-8')
            handle_incoming_message(message)

    handler = StreamHandler()
    
    # Corrected the subscribe_to_topic call to use topic and stream_handler as keyword arguments
    ipc_client.subscribe_to_topic(topic=TOPIC, stream_handler=handler)
    print("Subscribed successfully to topic")

# ---------- Message Handler ----------
def handle_incoming_message(message):
    print("Received MQTT message...")
    payload = json.loads(message)

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
            sqs_payload = {
                "request_id": request_id,
                "filename": file_name,
                "content": face_b64
            }

            # Send to SQS
            sqs_client.send_message(
                QueueUrl=SQS_QUEUE_URL,
                MessageBody=json.dumps(sqs_payload)
            )
            print(f"Face detected and message sent to SQS for request_id={request_id}")
        else:
            print("No face detected. Nothing sent to SQS.")

    except Exception as e:
        print(f"Error handling MQTT message: {e}")

# ---------- Main ----------
if __name__ == '__main__':
    subscribe_to_mqtt()
    print(f"Subscribed to topic {TOPIC}")
    while True:
        time.sleep(0.01)
