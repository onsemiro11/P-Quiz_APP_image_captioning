import sys
from flask import Flask, request, jsonify, send_file
import json
import base64
import image_captioning.generate_caption_DI as gc
import image_captioning.encoder_decoder_model_DIJ as encoder_decoder_model_DIJ

static_dir = 'app_back/image_captioning/data/images/'
data_path = 'app_back/data'

captions = encoder_decoder_model_DIJ.showtheresult(static_dir + 'sample.jpg')
# 모델 결과 dict 형태로 저장
model_result = {"captions": captions}
print(model_result)

