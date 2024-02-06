import yaml


def get_config(root, type_config):
	CONFIG = None
	if type_config == "facereg":
		# CONFIG = {"model_path": f"{str(root)}/weights/mxnet_regFace.onnx",
		# 			"imgsz": [112,112],
		# 			"conf_thres": 0.75,
		# 			"device": 'cpu'}

		config_face_regconize = f"{str(root)}/configs/arcface/face.yaml"
		with open(config_face_regconize, 'r') as stream:
		    CONFIG = yaml.safe_load(stream)
		    CONFIG["model_path"] = f"{str(root)}/weights/mxnet_regFace.onnx"

	if type_config == "facedet":
		config_retinanet_detFace = f"{str(root)}/configs/retinanet/detectFace.yaml"
		with open(config_retinanet_detFace, 'r') as stream:
		    CONFIG = yaml.safe_load(stream)
		    CONFIG["model_path"] = f"{str(root)}/weights/detectFace_model.pt"

	return CONFIG
