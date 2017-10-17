# -*- coding:utf-8 -*-
"""
File Name: dhb
Version:
Description:
Author: liuxuewen
Date: 2017/10/12 12:48
"""
from flask_restful import Resource
from flask import request, jsonify
import json
import base64

from crack_captcha.crack_cnn_tensorflow import predict


class DaiHouBang(Resource):
    def get(self):
        return {'hello': 'world'}

    def post(self):
        if request.headers['Content-Type'] == 'text/plain':
            return "Text Message: " + request.data

        elif request.headers['Content-Type'] == 'application/json':
            captcha = request.json['img']
            captcha = base64.b64decode(captcha)
            with open('p1.png', 'wb') as f:
                f.write(captcha)
            result = predict()
            return jsonify({'result': result})

            # return jsonify({'task': 'a'}), 201
