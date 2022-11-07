# pose-kepoint-api

ポーズ画像のキーポイント情報を取得できるツールのAPI

Lambda + API Gateway

API_URL: https://lddngv4k56.execute-api.ap-northeast-1.amazonaws.com/v2/pose-keypoint

## Request

```json
{
    "body": <画像をbase64にエンコードしたもの>,
    "model": <"lightning" or "thunder">,
    "format": <"normal" or "lite">
}
```

`format: "normal"` は下記(TODO)の問題により、使用不可。要調整。

## Response

```json
{
    "body": {
        "data": {
            "score": <全体のスコア>,
            "keypoints": <各キーポイントの情報（x,y座標,スコア）>
        }
    },
    "file": {
        "image": <640x640にリサイズしたキーポイント画像をエンコードしたもの>,
        "original_image": <元のアスペクト比のキーポイント画像をエンコードしたもの>,
    },
    "info": {
        "model": <推論で使用したモデル名>,
        "time": <推論時間>,
    }
}
```

`file: image` は元の画像のアスペクト比を保ったまま、その画像を640x640の黒塗り潰し画像ないに配置したもの。

## TODO 
- API GatewayのTimeOut上限が29秒
- Labmdaの実行で60秒近くかかることがある
- Lambdaを非同期に呼び出せるようにする