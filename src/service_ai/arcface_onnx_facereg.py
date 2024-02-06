import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.append(str(FILE.parents[1]))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from libs.base_libs import *
from libs.face_preprocess import preprocess as face_preprocess
from scipy import spatial
from sklearn import preprocessing

class FaceRegRunnable():

    def __init__(self, model_path, imgsz, conf_thres, device):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.imgsz = imgsz

        if str(device) in ['0', '1', '2', '3', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda']:
            providers = [("CUDAExecutionProvider", {"device_id": int(device)}), 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(model_path, providers=providers)

    def get_feature(self, ims, dets):
        if len(dets[0]["loc"])==0:
            return None

        outputs = []
        for i, det in enumerate(dets):
            # im = np.array(im.convert('RGB'))
            im = ims[i]
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            landms = det["landms"]
            bboxes = det["loc"]

            biggestBox = None
            maxArea = 0
            for j, bboxe in enumerate(bboxes):
                x1, y1, x2, y2 = bboxe
                area = (x2-x1) * (y2-y1)
                if area > maxArea and area > 900:
                # if area > maxArea:
                    maxArea = area
                    biggestBox = bboxe
                    landmarks = landms[j]
            if biggestBox is not None:
                bbox = np.array(biggestBox)
                landmarks = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
                            landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]])
                landmarks = landmarks.reshape((2,5)).T

                nimg = face_preprocess(im, bbox, landmarks, image_size=self.imgsz)
                nimg_transformed = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                nimg_transformed = np.transpose(nimg, (2,0,1))

                input_blob = np.expand_dims(nimg_transformed, axis=0)
                input = {}
                input["data"] = input_blob.astype(np.float32)
                embedding = self.sess.run(None, input)
                print(embedding[0].shape)
                embedding = preprocessing.normalize(embedding[0]).flatten()

                facenet_fingerprint = embedding.reshape(1,-1)
                # if (len(facenet_fingerprint) > 0):
                #     for value in facenet_fingerprint[0]:
                #         feature_str = feature_str + str(value) + "#"
                if len(outputs)==0:
                    outputs = facenet_fingerprint
                else:
                    outputs = np.concatenate((outputs,facenet_fingerprint), axis=0)
        return outputs

    def get_feature_without_det(self, ims):
        outputs = []
        for i, im in enumerate(ims):
            im = cv2.resize(im, (112,112), interpolation = cv2.INTER_AREA)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = np.transpose(im, (2,0,1))

            im_transformed = np.expand_dims(im, axis=0)
            input = {"data": im_transformed.astype(np.float32)}
            embedding = self.sess.run(None, input)
            embedding = preprocessing.normalize(embedding[0]).flatten()
            facenet_fingerprint = embedding.reshape(1,-1)
            if len(outputs)==0:
                outputs = facenet_fingerprint
            else:
                outputs = np.concatenate((outputs,facenet_fingerprint), axis=0)
        # print(outputs)
        print(outputs.shape)
        return outputs

    def compare_face_1_n_1(self, feature1, feature2):
        # print("---------------feature1: ", feature1)
        # print("---------------feature2: ", feature2)
        # dist = spatial.distance.euclidean(feature1[0], feature2[0])
        dist = np.linalg.norm(feature - feature1, axis=1)
        similarity = (np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2
        similarity_sort_idx = similarity.argsort()[::-1]
        return similarity, similarity_sort_idx[0]

    def compare_face_1_n_n(self, feature, features):
        dist = np.linalg.norm(feature - features, axis=2)
        # similarity = (np.tanh((1.22507105 - dist) * 7.321198934) + 1) / 2
        similarity = (np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2
        similarity = np.mean(similarity, axis=1)
        similarity_sort_idx = similarity.argsort()[::-1]
        return similarity, similarity_sort_idx
