import cv2
import torch
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from .common.facedetector import FaceDetector
from .train import MaskDetector

import time, asyncio, websockets, base64
import argparse

def tag_frame(frame, faceDetector, model, transformations, device):
    """
    Annotate a frame by detecting wether people in it are wearing mask
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ['No mask', 'Mask']
    labelColor = [(10, 0, 255), (10, 255, 0)]

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = faceDetector.detect(frame)
    for face in faces:
        xStart, yStart, width, height = face
        
        # clamp coordinates that are outside of the image
        xStart, yStart = max(xStart, 0), max(yStart, 0)
        
        # predict mask label on extracted face
        faceImg = frame[yStart:yStart+height, xStart:xStart+width]
        output = model(transformations(faceImg).unsqueeze(0).to(device))
        _, predicted = torch.max(output.data, 1)
        
        # draw face frame
        cv2.rectangle(frame,
                    (xStart, yStart),
                    (xStart + width, yStart + height),
                    (126, 65, 64),
                    thickness=2)
        
        # center text according to the face frame
        textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
        textX = xStart + width // 2 - textSize[0] // 2
        
        # draw prediction label
        cv2.putText(frame,
                    labels[predicted],
                    (textX, yStart-20),
                    font, 1, labelColor[predicted], 2)
    return frame


async def stream_cam(websocket_uri, model_path, fps):
    """
    Detect face masks on camera and stream it to websocket_uri
    """
    model = MaskDetector()
    model.load_state_dict(torch.load(model_path)['state_dict'], strict=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    faceDetector = FaceDetector(
        prototype='covid-mask-detector/models/deploy.prototxt.txt',
        model='covid-mask-detector/models/res10_300x300_ssd_iter_140000.caffemodel',
    )
    
    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])
    
    cam = cv2.VideoCapture(0)
    async with websockets.connect(websocket_uri) as websocket:
        while True:
            _, frame = cam.read()
            frame = tag_frame(frame, faceDetector, model, transformations, device)
            frame = cv2.imencode('.jpg', frame)[1]
            b64_img = base64.b64encode(frame)
            await websocket.send("data:image/jpeg;base64, " + b64_img.decode("utf-8"))
            print(f'Image sent to ws')
            time.sleep(1/fps)


# pylint: disable=no-value-for-parameter
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--websocket-uri', type=str, default='ws://localhost:3000',
                        help='Websocket server URI for streaming results.')
    parser.add_argument('-m', '--model-path', type=str, default='covid-mask-detector/models/face_mask.ckpt',
                        help='Path to PyTorch model to use for face mask detection.')
    parser.add_argument('-f', '--fps', type=int, default=5,
                        help='Number of frames per second to tag')
    args = parser.parse_args()
    asyncio.get_event_loop().run_until_complete(stream_cam(args.websocket_uri, args.model_path, args.fps))
