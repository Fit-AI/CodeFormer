import io
import os
import sys
import uuid
import time
import json
import shutil
import logging
logging.basicConfig(level=logging.DEBUG)

import httpx
import socket
import requests
import httpcore
from PIL import Image
from flask import (
        Flask,
        Response,
        request,
        jsonify
        )
import storage3
from supabase import (
        create_client,
        Client
        )

import cv2

from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

def storage_for_bucket(bucket:str):
    client : Client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
    return client.storage().get_bucket(bucket)

def upload_to_supabase(storage, key, target):
    assert key
    assert os.path.isfile(target)
    logging.debug(f'Uploading video {target} to {key}')

    MAX_TRIES = 10
    while MAX_TRIES >= 0:
        try:
            storage.upload(key, target, {'content-type': 'video/mp4'})
            return storage.create_signed_url(key, sys.maxsize)['signedURL']
        except storage3.utils.StorageException as e:
            code = e.args[0]['statusCode']
            message = e.args[0]['message']
            if code == 400 and message.find('duplicate key value') != -1:
                return storage.create_signed_url(key, sys.maxsize)['signedURL']
            else:
                logging.error(f'Uploading {target} -> {key} failed: [{e}]')
                continue
        except socket.timeout as e:
            logging.error(f'Timed out! {e}')
            MAX_TRIES -= 1
            time.sleep(1)
            continue
        except httpx.ReadTimeout as e:
            logging.error(f'Supabase timed out uploading! {e}')
            MAX_TRIES -= 1
            time.sleep(1)
            continue
        except httpcore.ReadTimeout as e:
            logging.error(f'Other timed out?? {e}')
            MAX_TRIES -= 1
            time.sleep(1)
            continue

class Upscaler(object):

    INPUT_VIDEO = 'inputs/input_video.mp4'
    TEMP_PATH = 'temp/input_video.mp4'

    def __init__(self):
        # Create sim-swap
        self.model = create_model(self.opt)
        self.model.eval()
        # Create face-detection
        self.app = Face_detect_crop(name='antelope', root='./insightface_func/models')
        self.app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)

    def codeformer(self)-> dict:
        assert os.path.isfile(Upscaler.INPUT_VIDEO)
        shutil.rmtree('temp', ignore_errors=True)
        os.makedirs('temp')
        shutil.rmtree('outputs', ignore_errors=True)
        os.makedirs('outputs')

        with torch.no_grad():
            pic_a = Upscaler.INPUT_IMAGE
            assert os.path.isfile(pic_a)
            img_a = Image.open(pic_a).convert('RGB')
            img_a_whole = cv2.imread(pic_a)
            img_a_align_crop, _ = self.app.get(img_a_whole, self.crop_size)
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
            img_a = transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
            img_id = img_id.cuda()
            # Latent ID
            img_id_downsample = F.interpolate(img_id, size=(112,112))
            latend_id = self.model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)
            # Swap
            video_swap(self.opt.video_path,
                       latend_id,
                       self.model,
                       self.app,
                       self.opt.output_path,
                       temp_results_dir=self.opt.temp_path,
                       no_simswaplogo=self.opt.no_simswaplogo,
                       use_mask=self.opt.use_mask,
                       crop_size=self.crop_size)

            storage = storage_for_bucket('dev-codeformer')
            key = str(uuid.uuid4())
            result_url = upload_to_supabase(storage,
                                            key,
                                            'results/result.mp4')
            return {'result': result_url}

def download_video(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

swapper = Upscaler()

app = Flask('codeformer-api')

@app.route('/livez', methods=['GET', 'POST'])
def livez():
    return Response("{}", status=200)

@app.route('/codeformer', methods=['GET', 'POST'])
def codeformer():
    content_type = request.headers.get('Content-Type')
    video_url = request.json['video']
    logging.info(f'Running codeformer {target_url} on {video_url}')
    shutil.rmtree('inputs', ignore_errors=True)
    os.makedirs('inputs')
    shutil.rmtree('results', ignore_errors=True)
    os.makedirs('results')
    download_video(video_url, Upscaler.INPUT_VIDEO)
    return swapper.codeformer()
