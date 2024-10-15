import os
import random
import cv2
from ultralytics import YOLO
from deepsort_tracker import Tracker
from aiogram import Bot, Dispatcher, types
import asyncio
import time
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
bot = Bot(token=TOKEN)
dp = Dispatcher()


cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

model = YOLO("yolov8n.pt")
model.fuse()

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5
tracked_objects = {}

async def send_alert(frame, track_id):
    photo = cv2.imencode('.jpg', frame)[1].tobytes()
    await bot.send_photo(chat_id=CHAT_ID, photo=photo)
    print(f"Alert: Object {track_id} detected for more than 3 seconds.")

async def main():
    global ret, frame
    while ret:
        results = model(frame, verbose=False)

        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1, y2, x1, y2, class_id = map(int, (x1, y1, x2, y2, class_id))
                if score > detection_threshold:
                    detections.append([x1, y1, x2, y2, score])

            tracker.update(frame, detections)

            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

                if track_id in tracked_objects:
                    tracked_objects[track_id]['time'] += 1 / cap.get(cv2.CAP_PROP_FPS)
                else:
                    tracked_objects[track_id] = {'time': 0}

                if tracked_objects[track_id]['time'] > 3:
                    await send_alert(frame, track_id)
                    tracked_objects[track_id]['time'] = 0  

        cap_out.write(frame)
        ret, frame = cap.read()

    cap.release()
    cap_out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())