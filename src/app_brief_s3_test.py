import os, sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
# WEIGHTS = ROOT / 'weights'
DB_FT = "static/database_features"
PATH_DB_FT = f"{str(ROOT)}/{DB_FT}"
if not os.path.exists(PATH_DB_FT):
	os.mkdir(PATH_DB_FT)

IMG_AVATAR = "static/avatar"
PATH_IMG_AVATAR = f"{str(ROOT)}/{IMG_AVATAR}"
if not os.path.exists(PATH_IMG_AVATAR):
	os.mkdir(PATH_IMG_AVATAR)

from flask            import Flask, session, Blueprint, json
from flask_restx import Resource, Api, fields, inputs
from flask_cors 	  import CORS
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from rq import Queue
from rq.command import send_stop_job_command
from rq.job import Job
from worker import conn
from werkzeug.datastructures import FileStorage
#from track_briefcam_customv2 import track_briefcam, delete_output, PATH_LOG_ABSOLUTE
# from search_vehicle import search_vehicle
#from merge_video import get_merge_video
import socket

from service_ai.arcface_onnx_facereg import FaceRegRunnable
from service_ai.retinanet_det import RetinanetRunnable
from configs.base_config import get_config
import io
from PIL import Image
import cv2
import numpy as np
import time
import shutil

SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
# print(SQLALCHEMY_DATABASE_URI)

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
# print(s.getsockname()[0])

app = Flask(__name__)

app.config.from_object('configuration.ConfigMYSQL')
if SQLALCHEMY_DATABASE_URI is not None:
	app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
# print(app.config)

db = SQLAlchemy(app) # flask-sqlalchemy
migrate = Migrate(app, db)
q = Queue(connection=conn)
api_bp = Blueprint("api", __name__, url_prefix="/api")

api = Api(api_bp, version='1.0', title='Brief Cam API',
	description='Brief Cam API for everyone', base_url='/api'
)
app.register_blueprint(api_bp)
CORS(app, supports_credentials=True, allow_headers=['Content-Type', 'X-ACCESS_TOKEN', 'Authorization'], origins=[f"http://{s.getsockname()[0]}:3456"])


#-----------------------models AI-------------------------
CONFIG_FACEREG = get_config(root=ROOT, type_config="facereg")
facereg = FaceRegRunnable(**CONFIG_FACEREG)

CONFIG_FACEDET = get_config(root=ROOT, type_config="facedet")
facedet = RetinanetRunnable(**CONFIG_FACEDET)
