import os
# import subprocess

# print(os.path.abspath('./movenet_singlepose_thunder_4/saved_model.pb'))
# print(os.getcwd())
# res = subprocess.check_call(['tree','{}'.format(os.getcwd())])

# Import TF and TF Hub libraries.
import tensorflow as tf
import numpy as np
import cv2
import json
import base64
import time

# 外部モジュール
import resize

infer_time = time.time()

# Lambda Function
def handler(event, context):
    """
    API種類(モデルrtサイズ):
    - /lightning/ : 低レイテンシ向け(20.6MB)
    - /lightning/lite : 軽量モデル(4.8MB)
    - /thunder/ : 高精度向け(36.4MB)
    - /thunder/lite : 軽量モデル(12.6MB)
    """
    global infer_time
    infer_time = time.time()
    # lightning or thunder
    contents = json.loads(event["body"])
    model = contents.get('model')
    # normal(TF) or lite(TFLite_float16)
    format = contents.get('format')

    print(os.getcwd())

    if event.get('body') == None or model == None or format == None:
        print("wrong request parameters.")
    elif model == "lightning":
        if format == "lite":
            print("Model: lightning / Format: lite.")
            return keypoint_lightning_TFLite(event, context)
        else:
            print("Model: lightning / Format: normal.")
            return keypoint_lightning_TF(event, context)
    elif model == "thunder":
        if format == "lite":
            print("Model: thunder / Format: lite.")
            return keypoint_thunder_TFLite(event, context)
        else:
            print("Model: thunder / Format: normal.")
            return keypoint_thunder_TF(event, context)
    else:
        print("No matching model exist.")

    return {
        "isBase64Encoded": False,
        "statusCode": 400,
        "headers": {
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST'
        },
        "body": "Wrong request parameters."
    }


# 各モデルでの推論実行
# モデルを読み込んで渡す
## thunder, TF
def keypoint_thunder_TF(event, context):
    model = tf.saved_model.load(os.getcwd()+"/movenet_singlepose_thunder_4")
    event['model_name'] = "movenet_singlepose_thunder_4"
    return keypoint_use_TF(event, context, model, 256)

## thunder, Lite
def keypoint_thunder_TFLite(event, context):
    interpreter = tf.lite.Interpreter(model_path=os.getcwd()+"/lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite")
    event['model_name'] = "movenet_singlepose_thunder_tflite_float16_4"
    return keypoint_use_TFLite(event, context, interpreter, 256)

## lightning, TF
def keypoint_lightning_TF(event, context):
    model = tf.saved_model.load(os.getcwd()+"/movenet_singlepose_lightning_4")
    event['model_name'] = "movenet_singlepose_lightning_4"
    return keypoint_use_TF(event, context, model, 192)

## lightning, Lite
def keypoint_lightning_TFLite(event, context):
    interpreter = tf.lite.Interpreter(model_path=os.getcwd()+"/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite")
    event['model_name'] = "movenet_singlepose_lightning_tflite_float16_4"
    return keypoint_use_TFLite(event, context, interpreter, 192)


# 画像を推論
## TFライブラリを使った推論
def keypoint_use_TF(event, context, model, input_size):
    # Initialize the TF model
    movenet = model.signatures['serving_default']

    #
    contents = json.loads(event["body"])
    img_binary = base64.b64decode(contents["body"])
    jpg = np.frombuffer(img_binary,dtype=np.uint8)
    print(jpg.size)
    #
    dst_img, scale, h, w = resize.resize(jpg, input_size, input_size) # リサイズ
    # dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)  # BGR→RGB変換
    # dst_img = dst_img.reshape(-1, input_size, input_size, 3)  # リシェイプ
    dst_img, h, w = resize.scale_to_height(dst_img, 640, h, w)

    print(dst_img.shape)
    print(h, w)

    image = tf.expand_dims(dst_img, axis=0)
    image = tf.image.resize_with_pad(image, input_size, input_size)
    input_image = tf.cast(image, dtype=tf.int32)  # int32へキャスト

    print("image scale: {}".format(scale))

    # Run model inference.
    outputs = movenet(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']

    return response_keypoint_info(keypoints.numpy(), dst_img, h, w, event)

## TFLiteライブラリを使った推論
def keypoint_use_TFLite(event, context, interpreter, input_size):
    # Initialize the TFLite interpreter
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    ##
    contents = json.loads(event["body"])
    img_binary = base64.b64decode(contents["body"])
    jpg = np.frombuffer(img_binary,dtype=np.uint8)
    print(jpg.size)
    # scale = resize.resize_test(jpg, 'resize_pose.jpg', 192, 192)
    dst_img, scale, h, w = resize.resize(jpg, input_size, input_size)
    dst_img, h, w = resize.scale_to_height(dst_img, 640, h, w)
    print("image scale: {}".format(scale))

    # Load the input image.
    # image_path = 'resize_pose.jpg'
    # image = tf.io.read_file(image_path)
    # image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(dst_img, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.image.resize_with_pad(image, input_size, input_size)

    # TF Lite format expects tensor type of float32. uint8
    input_image = tf.cast(image, dtype=tf.uint8)
    # input_image = tf.cast(image, dtype=tf.float32)

    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())

    interpreter.invoke()

    # print("time invoke: {}".format(time.time() - time_start))

    # Output is a [1, 1, 17, 3] tensor.
    keypoints = interpreter.get_tensor(output_details[0]['index'])

    return response_keypoint_info(keypoints, dst_img, h, w, event)


# レスポンスの作成
def response_keypoint_info(keypoints, dst_img, h, w, event):
    # js用にオブジェクト形式で出力する
    name = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"]

    # 配列に変換
    keypoints_with_scores = np.squeeze(keypoints)

    score_avg = float(sum(keypoints_with_scores[:,2])/len(name))
    keypoints_list = []
    for i, key in enumerate(keypoints_with_scores):
        # coordinate: key[0]:y, key[1]:x
        keypoints_list.append({
            "y": float(key[0]), "x": float(key[1]), "score": float(key[2]), "name": name[i]
        })

    output = {
        "score": score_avg,
        "keypoints": keypoints_list
    }

    json_data = [];
    json_data.append(output)

    # print(json_data) 

    # JSONファイルに出力
    data = {
        "data": json_data,
    }
    
    print("Successed infer.")


    # キーポイント情報付き画像を取得
    sample_image = get_keypoint_img(dst_img, keypoints_with_scores)
    print(sample_image.shape)
    sample_image_original = sample_image[0:h, 0:w]
    print(sample_image_original.shape)

    keypoint_image_bytes = cv2.imencode('.jpg', sample_image)[1].tobytes()
    keypoint_image = base64.b64encode(keypoint_image_bytes)

    keypoint_original_image_bytes = cv2.imencode('.jpg', sample_image_original)[1].tobytes()
    keypoint_original_image = base64.b64encode(keypoint_original_image_bytes)

    res = {
        "body": data,
        "file": {
            "image": keypoint_image.decode('utf-8'),
            "original_image": keypoint_original_image.decode('utf-8'),
        },
        "info": {
            "model": event.get('model_name'),
            "time": float("{:.3f}".format(time.time() - infer_time)),
        },
    }

    return {
        "isBase64Encoded": True,
        "statusCode": 200,
        "headers": {
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST'
        },
        "body": json.dumps(res, indent=4, ensure_ascii=False)
    }
    

# 可視化
def get_keypoint_img(dst_img, keypoints_with_scores):
    """
    ポーズ画像を受け取って、キーポイント情報可視化した画像を返す関数
    """
    sample_image = dst_img
    image_width, image_height = sample_image.shape[1], sample_image.shape[0]

    # 点位置を保存
    skeleton_points = []
    for i in range(len(keypoints_with_scores)):
        # 推論値のXY座標は相対値のため、画像サイズを乗算し絶対座標にする
        keypoint_x = int(image_width * keypoints_with_scores[i][1])
        keypoint_y = int(image_height * keypoints_with_scores[i][0])
        # スコア
        score = keypoints_with_scores[i][2]

        skeleton_points.append([keypoint_x, keypoint_y, score])

        # スコアが0.3以上のキーポイントを可視化する
        if score > 0.3:
            cv2.circle(sample_image, (keypoint_x, keypoint_y), 3, (0, 255, 0), 1)
            cv2.putText(sample_image, str(i), (keypoint_x, keypoint_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)

    # 線
    connect_list = [
        [0, 1, (255, 0, 0)],  # 鼻 → 左目
        [0, 2, (0, 0, 255)],  # 鼻 → 右目
        [1, 3, (255, 0, 0)],  # 左目 → 左耳
        [2, 4, (0, 0, 255)],  # 右目 → 右耳
        # [0, 5, (255, 0, 0)],  # 鼻 → 左肩
        # [0, 6, (0, 0, 255)],  # 鼻 → 右肩
        [5, 6, (0, 255, 255)],  # 左肩 → 右肩
        [5, 7, (255, 0, 0)],  # 左肩 → 左肘
        [7, 9, (255, 0, 0)],  # 左肘 → 左手首
        [6, 8, (0, 0, 255)],  # 右肩 → 右肘
        [8, 10, (0, 0, 255)],  # 右肘 → 右手首
        [11, 12, (0, 255, 255)],  # 左尻 → 右尻
        [5, 11, (255, 0, 0)],  # 左肩 → 左尻
        [11, 13, (255, 0, 0)],  # 左尻 → 左膝
        [13, 15, (255, 0, 0)],  # 左膝 → 左足首
        [6, 12, (0, 0, 255)],  # 右肩 → 右尻
        [12, 14, (0, 0, 255)],  # 右尻 → 右膝
        [14, 16, (0, 0, 255)],  # 右膝 → 右足首
    ]
    for i in range(len(connect_list)):
        p1, p2, color = connect_list[i]
        x1, y1, score1 = skeleton_points[p1]
        x2, y2, score2 = skeleton_points[p2]

        if score1 > 0.3 and score2 > 0.3:
            cv2.line(sample_image, (x1, y1), (x2, y2), color, 1)

    return sample_image


# デバッグ用
def test(event, context):
    print("This is test")
    res = {
        "test": "TESTTEST"
    }
    return {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": {},
        "body": json.dumps(res, indent=4, ensure_ascii=False)
    }
def debug(event, context):
    print(event)
    return {
        "isBase64Encoded": True,
        "statusCode": 200,
        "headers": {},
        "body": json.dumps(event, indent=4, ensure_ascii=False)
    }


# デバッグ用実行スクリプト
if __name__ == '__main__':
    with open("../lib/pose_sample/pose20.jpg", "rb") as image_file:
        data = base64.b64encode(image_file.read())

    with open('bynary.txt', 'w') as f:
        f.write(data.decode('utf-8'))


    req = {
        "body": data.decode('utf-8'),
        "model": "lightning",
        "format": "normal",
    }
    event = {
        "body": json.dumps(req, indent=4, ensure_ascii=False)
    }


    ### after post api

    response = handler(event, None)
    # response = debug(event, None)
    # response = test(event, None)
    # response = convert(event, None)
    
    print(response)

    contents = json.loads(response["body"])
    data = contents["body"]["data"]
    file = contents["file"]
    info = contents["info"]
    # print(file)

    # バイナリ <- 文字列
    ## 黒枠あり
    image_binary = base64.b64decode(file["image"])
    # print(type(image_binary))

    # jpg = np.frombuffer(image_binary,dtype=np.int32)
    jpg = np.frombuffer(image_binary,dtype=np.uint8)
    print(jpg.size)

    scale = resize.resize_test(jpg, 'keypoints_scaled.jpg', 640, 640)
    print(scale)

    ## 黒枠なし
    image_binary = base64.b64decode(file["original_image"])

    # jpg = np.frombuffer(image_binary,dtype=np.int32)
    jpg = np.frombuffer(image_binary,dtype=np.uint8)
    print(jpg.size)

    scale = resize.resize_test(jpg, 'keypoints_original.jpg', 640, 640)
    print(scale)

    print(info)