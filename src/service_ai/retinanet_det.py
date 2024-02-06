import sys
# import numpy as np
# import bentoml
# import cv2
# import torch 
from math import ceil
from itertools import product as product

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.append(str(FILE.parents[1]))
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))

from libs.yolo_utils import select_device
from libs.base_libs import *

class Dets(BaseModel):
	loc: List[list] = []
	landms: List[list] = []

class PriorBox(object):
	def __init__(self, min_sizes=[[16,32],[64,128],[256,512]], steps=[8,16,32], clip=False, image_size=None, phase='train'):
		super(PriorBox, self).__init__()
		self.min_sizes = min_sizes
		self.steps = steps
		self.clip = clip
		self.image_size = image_size
		self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
		self.name = "s"

	def forward(self):
		anchors = []
		for k, f in enumerate(self.feature_maps):
			min_sizes = self.min_sizes[k]
			for i, j in product(range(f[0]), range(f[1])):
				for min_size in min_sizes:
					s_kx = min_size / self.image_size[1]
					s_ky = min_size / self.image_size[0]
					dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
					dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
					for cy, cx in product(dense_cy, dense_cx):
						anchors += [cx, cy, s_kx, s_ky]

		# back to torch land
		output = torch.Tensor(anchors).view(-1, 4)
		if self.clip:
			output.clamp_(max=1, min=0)
		return output

class RetinanetRunnable():
	def __init__(self, model_path, min_sizes, steps, variance, clip, conf_thres, iou_thres, device):
		self.min_sizes = min_sizes
		self.steps = steps
		self.variance = variance
		self.clip = clip
		self.conf_thres = conf_thres
		self.iou_thres = iou_thres
		# if str(device) in ['0', '1', '2', '3']:
		#     self.device = f"cuda:{int(device)}"
		# elif str(device) in ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']:
		# 	self.device = str(device)
		# else:
		#     self.device = "cpu"
		self.device = select_device(device)
		self.model = torch.load(model_path, map_location=self.device)

	def preProcess(self, img):
		img = np.float32(img)
		im_height, im_width, _ = img.shape
		scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
		img -= (104, 117, 123)
		img = img.transpose(2, 0, 1)
		img = torch.from_numpy(img).unsqueeze(0)
		img = img.to(self.device)
		scale = scale.to(self.device)
		return [img, scale, im_height, im_width]

	def postProcess(self, input, output):
		img, scale, im_height, im_width = input
		loc, conf, landms = output
		priorbox = PriorBox(min_sizes=self.min_sizes, steps=self.steps, \
							clip=self.clip, image_size=(im_height, im_width))
		priors = priorbox.forward()
		priors = priors.to(self.device)
		prior_data = priors.data
		boxes = self.decode(loc.data.squeeze(0), prior_data, self.variance)
		boxes = boxes * scale
		boxes = boxes.cpu().numpy()
		scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
		landms = self.decode_landm(landms.data.squeeze(0), prior_data, self.variance)
		scale_landms = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
							   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
							   img.shape[3], img.shape[2]])
		scale_landms = scale_landms.to(self.device)
		landms = landms * scale_landms
		landms = landms.cpu().numpy()

		inds = np.where(scores > self.conf_thres)[0]
		boxes = boxes[inds]
		landms = landms[inds]
		scores = scores[inds]

		order = scores.argsort()[::-1][:5000]
		boxes = boxes[order]
		landms = landms[order]
		scores = scores[order]

		dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
		keep = self.py_cpu_nms(dets, self.iou_thres)
		dets = dets[keep, :]
		landms = landms[keep]

		dets = dets[:750, :]
		landms = landms[:750, :]
		dets = np.concatenate((dets, landms), axis=1)
		return dets

	def py_cpu_nms(self, dets, thresh):
		x1 = dets[:, 0]
		y1 = dets[:, 1]
		x2 = dets[:, 2]
		y2 = dets[:, 3]
		scores = dets[:, 4]

		areas = (x2 - x1 + 1) * (y2 - y1 + 1)
		order = scores.argsort()[::-1]

		keep = []
		while order.size > 0:
			i = order[0]
			keep.append(i)
			xx1 = np.maximum(x1[i], x1[order[1:]])
			yy1 = np.maximum(y1[i], y1[order[1:]])
			xx2 = np.minimum(x2[i], x2[order[1:]])
			yy2 = np.minimum(y2[i], y2[order[1:]])

			w = np.maximum(0.0, xx2 - xx1 + 1)
			h = np.maximum(0.0, yy2 - yy1 + 1)
			inter = w * h
			ovr = inter / (areas[i] + areas[order[1:]] - inter)

			inds = np.where(ovr <= thresh)[0]
			order = order[inds + 1]

		return keep


	def decode(self, loc, priors, variances):
		boxes = torch.cat((
			priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
			priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
		boxes[:, :2] -= boxes[:, 2:] / 2
		boxes[:, 2:] += boxes[:, :2]
		return boxes

	def decode_landm(self, pre, priors, variances):
		landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
							priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
							priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
							priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
							priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
							), dim=1)
		return landms

	def inference(self, ims):
		results = []
		miss_det = []
		for i, im in enumerate(ims):
			# im = np.array(im.convert('RGB'))
			im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
			input = self.preProcess(im)
			img = input[0]
			loc, conf, landms = self.model(img)
			output = [loc, conf, landms]
			dets = self.postProcess(input, output)
			if len(dets) != 0:
				result = dict(loc=dets[:,:4], conf=dets[:,4], landms=dets[:,5:])
				results.append(result)
			else:
				miss_det.append(i+1)
		# print(results)
		return results, miss_det

	def render(self, ims):
		im_preds = []
		for i, im in enumerate(ims):
			# im = np.array(im.convert('RGB'))
			im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
			input = self.preProcess(im)
			img = input[0]
			loc, conf, landms = self.model(img)
			output = [loc, conf, landms]
			dets = self.postProcess(input, output)
			for det in dets:
				cv2.rectangle(im, det[:2].astype(int), det[2:4].astype(int), (255, 0, 0), 2)
			im_preds.append(im)
		return im_preds