import sys
from flask import Flask, request, jsonify, send_file
import json
import base64

import image_captioning.encoder_decoder_model_DIJ as encoder_decoder_model_DIJ
import image_captioning.generate_caption as vg

#sys.path.insert(0, '/Users/hyundolee/campus_project/app/pquiz_app')

app = Flask(__name__)
static_dir = 'image_captioning/data/images/'
data_path = 'data'


@app.route('/api', methods=['GET', 'POST'])
def model():
    r = request.method
    # front에 캡션과 이미지 json 형식으로 전달
    if r == 'GET':
        with open(data_path + "/data.json") as f:
            data = json.load(f)
        return data
    # front에서 이미지 받아와 예측 모델 구동
    elif r == 'POST':
        print('gggggg')
        # base64 형태로 인코딩 되어 있는 이미지 디코딩 후 static_dir에 저장
        with open(static_dir + 'sample.jpg', "wb") as fh:
            fh.write(base64.decodebytes(request.data))
        # image captioning 모델 구동, list 형태로 나온 결과 string으로 변환
        captions = encoder_decoder_model_DIJ.showtheresult(static_dir + 'sample.jpg')
#        captions = vg.generate_caption(static_dir+'sample.jpg')
        # 모델 결과 dict 형태로 저장
        model_result = {"captions": captions}
        print(model_result)
        # front에 전달하기 위해 dict 형태에서 json object 형태로 변환
        with open(data_path + "/data.json", "w") as fjson:
            json_result = json.dumps(model_result)
            fjson.write(json_result)
        return jsonify({
            "captions": "Detected !"})
    else:
        return jsonify({
            "captions": "Refresh again !"
        })


@app.route('/result')
def sendimage():
    # 결과 이미지를 보기 위한 url
    return send_file(static_dir + 'sample.jpg', mimetype='image/gif')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=500)
