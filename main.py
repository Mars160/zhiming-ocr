from paddleocr import PaddleOCR
from fastapi import FastAPI, UploadFile
from threading import Thread
import os
import uuid

app = FastAPI()
model = PaddleOCR(use_angle_cls=True, use_gpu=False)
OCR_DICT = {}
response_base = {
    "code": 0,
    "data": None,
    "msg": "success"
}
TEMP_DIR = 'tmp-img'

if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)


def ocr(img):
    ocr_result = model.ocr(img, cls=True)[0]
    result = []
    for i in ocr_result:
        result.append(i[1][0])
    return result


def start_ocr(filename):
    filepath = '{}{}{}'.format(TEMP_DIR, os.sep, filename)
    OCR_DICT[filename] = 0
    result = ocr(filepath)
    OCR_DICT[filename] = result
    os.remove(filepath)


@app.get('/ocr/{uuid}')
async def getocr(uuid):
    response = response_base.copy()
    if uuid in OCR_DICT:
        if OCR_DICT[uuid] != 0:
            response['data'] = OCR_DICT[uuid]
            del OCR_DICT[uuid]
        else:
            response['data'] = None
            response['msg'] = '正在识别中'
    else:
        response['code'] = 404
        response['msg'] = 'OCR任务不存在'
    return response


@app.post('/ocr')
async def postocr(file: UploadFile):
    response = response_base.copy()
    contents = await file.read()
    uuid_s = uuid.uuid4().hex
    with open('{}{}{}'.format(TEMP_DIR, os.sep, uuid_s), 'wb') as f:
        f.write(contents)

    t = Thread(target=start_ocr, args=(uuid_s,))
    t.start()
    response['data'] = uuid_s
    return response
