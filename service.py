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
import ffmpeg
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
from supabase import (
        create_client,
        Client
        )
import storage3

from tqdm import tqdm

import cv2
import torch
assert torch.cuda.is_available()
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

def set_realesrgan():
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer
    model = RRDBNet(num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2)
    bg_tile = 400

    bg_upsampler = RealESRGANer(
        scale=2,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        model=model,
        tile=bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=True)  # need to set False in CPU mode
    return bg_upsampler

class Upscaler(object):

    INPUT_VIDEO = 'inputs/input_video.mp4'
    RESULTS_PATH = 'results/'
    TEMP_PATH = 'temp/'

    def __init__(self):
        self.upsampler = set_realesrgan()

        self.device = torch.device('cuda')
        self.net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                                   connect_list=['32', '64', '128', '256']).to(self.device)
        ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'],
                                        model_dir='weights/CodeFormer', progress=True, file_name=None)
        checkpoint = torch.load(ckpt_path)['params_ema']
        self.net.load_state_dict(checkpoint)
        self.net.eval()

        UPSCALE = 2
        self.face_helper = FaceRestoreHelper(UPSCALE,
                                        face_size=512,
                                        crop_ratio=(1, 1),
                                        det_model='retinaface_resnet50',
                                        save_ext='png',
                                        use_parse=True,
                                        device=self.device)
        logging.info('Initialized')

    def codeformer(self, w=0.9, max_frames=sys.maxsize, upscale_background=False)-> dict:
        assert os.path.isfile(Upscaler.INPUT_VIDEO)
        shutil.rmtree(Upscaler.TEMP_PATH, ignore_errors=True)
        os.makedirs(Upscaler.TEMP_PATH)

        video = cv2.VideoCapture(Upscaler.INPUT_VIDEO)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)

        print (f'[{frame_count}] frames @ [{fps}] fps')

        os.makedirs(os.path.join(Upscaler.RESULTS_PATH, 'cropped_faces'))
        os.makedirs(os.path.join(Upscaler.RESULTS_PATH, 'restored_faces'))
        os.makedirs(os.path.join(Upscaler.RESULTS_PATH, 'final_results'))

        for frame_index in tqdm(range(frame_count)):
            self.face_helper.clean_all()
            if frame_index > max_frames:
                logging.info(f'Breaking after {frame_index} frames')
                break
            ret, img = video.read()
            if not ret:
                logging.error(f'Failed to decode {frame_index}')
                break

            # cv2.imwrite(f'temp/{frame_index:06d}.png', img)
            self.face_helper.read_image(img)
            only_center_face=False
            num_det_faces = self.face_helper.get_face_landmarks_5(only_center_face=only_center_face,
                                                                  resize=640,
                                                                  eye_dist_threshold=5)
            self.face_helper.align_warp_face()

            for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
                cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
                try:
                    with torch.no_grad():
                        output = self.net(cropped_face_t, w=w, adain=True)[0]
                        restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                    del output
                    torch.cuda.empty_cache()
                except Exception as error:
                    print(f'\tFailed inference for CodeFormer: {error}')
                    restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                restored_face = restored_face.astype('uint8')
                self.face_helper.add_restored_face(restored_face)

            self.face_helper.get_inverse_affine(None)
            # bg_img = self.upsampler.enhance(img, outscale=2)[0]
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=None,
                                                                       draw_box=False,
                                                                       face_upsampler=self.upsampler)
            basename = 'input_video'
            save_intermediates = False
            if save_intermediates:
                for idx, (cropped_face, restored_face) in enumerate(zip(self.face_helper.cropped_faces, self.face_helper.restored_faces)):
                    save_crop_path = os.path.join(Upscaler.RESULTS_PATH,
                                                  'cropped_faces',
                                                  f'{basename}_{frame_index:02d}_{idx:02d}.png')
                    cv2.imwrite(save_crop_path, cropped_face)
                    save_face_name = f'{basename}_{frame_index:02d}_{idx:02d}.png'
                    save_restore_path = os.path.join(Upscaler.RESULTS_PATH,
                                                     'restored_faces',
                                                     save_face_name)
                    cv2.imwrite(save_restore_path, restored_face)

            # Save result
            save_restore_path = os.path.join(Upscaler.RESULTS_PATH,
                                             'final_results',
                                             f'{basename}_{frame_index:06d}.png')
            cv2.imwrite(save_restore_path, restored_img)

        FINAL_RESULTS_PATH = os.path.join(Upscaler.RESULTS_PATH, 'final_results')
        ffmpeg.input(f'{FINAL_RESULTS_PATH}/*.png', pattern_type='glob', framerate=fps).output(f'{Upscaler.RESULTS_PATH}/result.mp4', **{'qscale:v':0, 'c:v':'mpeg4'}).run()
        storage = storage_for_bucket('dev-codeformer')
        result_url = upload_to_supabase(storage,
                                        str(uuid.uuid4()),
                                        'results/result.mp4')
        return [{'result': result_url,
                 'parameters': {'w':
                                w}}]

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
    instances = request.json['instances']
    instance = instances[0]
    # Parameters
    video_url = instance['video']
    w = instance.get('w', 0.5)
    upscale = instance.get('upscale', 1)
    max_frames = instance.get('max_frames', sys.maxsize)
    upscale_background = instance.get('upscale_background', False)
    shutil.rmtree('inputs', ignore_errors=True)
    os.makedirs('inputs')
    shutil.rmtree(Upscaler.RESULTS_PATH, ignore_errors=True)
    os.makedirs(Upscaler.RESULTS_PATH)
    download_video(video_url, Upscaler.INPUT_VIDEO)
    logging.info(f'Running codeformer {video_url}')
    return jsonify({'predictions': swapper.codeformer(w=w, max_frames=max_frames, upscale_background=upscale_background)})
