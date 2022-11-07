import base64
import json
import requests
import app.resize as resize
import cv2
import numpy as np


def format_for_json(obj):
    encoded = base64.b64encode(obj)
    decoded = encoded.decode('utf-8')
    return decoded

with open("./lib/pose_sample/pose21.jpg", "rb") as f:
    data = base64.b64encode(f.read())
# data = "aaa"

data = {"body":data.decode('utf-8')}
response = requests.post("https://lddngv4k56.execute-api.ap-northeast-1.amazonaws.com/dev", json=data)

contents = json.loads(response.content)
data = contents["body"]["data"]
file = contents["file"]
# print(file)

# バイナリ <- 文字列
image_binary = base64.b64decode(file)
# print(type(image_binary))
# Image.open(io.BytesIO(image_binary))

jpg = np.frombuffer(image_binary,dtype=np.uint8)
print(jpg.size)
# # バイナリーストリーム <- バリナリデータ
# img_binarystream = io.BytesIO(image_binary)
# print(type(img_binarystream))

# # PILイメージ <- バイナリーストリーム
# img_pil = Image.open(img_binarystream)
# # print(img_pil.mode) #この段階だとRGBA

# # numpy配列(RGBA) <- PILイメージ
# img_numpy = np.asarray(img_pil)

# # umpy配列(BGR) <- numpy配列(RGBA)
# img_numpy_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGBA2BGR)

# dst_img = resize.scale_to_height(jpg, 640)
scale = resize.resize_test(jpg, 'keypoints.jpg', 640, 640)
print(scale)
# cv2.imwrite("keypoint_pose.jpg", dst_img)

# print(contents)
print(json.dumps(data, indent=4, ensure_ascii=False))