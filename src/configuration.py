import os 
from pathlib import Path 
from datetime import timedelta

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

class ConfigMYSQL():
	CSRF_ENABLED = True
	SECRET_KEY   = "77tgFCdrEEdv77554##@3"
	SQLALCHEMY_TRACK_MODIFICATIONS 	= True
	# SQLALCHEMY_DATABASE_URI = 'mysql://root:123456@localhost/facedb'
	# SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://vms_op:MQ123456@192.168.6.161:5433/vms_hcm'

	file_allowed = ['image/png', 'image/jpeg', 'application/octet-stream']
	JWT_EXPIRATION_DELTA = timedelta(days=365)
	SQLALCHEMY_POOL_SIZE = 100
	SQLALCHEMY_MAX_OVERFLOW = 100

	SQLALCHEMY_ENGINE_OPTIONS = {
									'pool_size': 100,
									"max_overflow": 200,
								}

# export DATABASE_URL=postgresql+psycopg2://vms_op:MQ123456@192.168.6.161:5433/vms_hcm
