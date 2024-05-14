from app_brief_s3_test import *
from modeldbs import User, CodeList, User2

upload_parser_faceRegister = api.parser()
upload_parser_faceRegister.add_argument("code", type=str, required=True)
upload_parser_faceRegister.add_argument("name", type=str, required=True)
upload_parser_faceRegister.add_argument("birthday", type=str, required=True)
upload_parser_faceRegister.add_argument("images", location='files', type=FileStorage, required=True, action='append')
@api.expect(upload_parser_faceRegister)
@api.route('/registerFace')
class registerFace(Resource):
	def post(self):
		try:
			args = upload_parser_faceRegister.parse_args()
			code = args["code"]
			name = args["name"]
			birthday = args["birthday"]

			codes = db.session.query(User.code).all()
			codes = list(chain(*codes))
			# print(codes)
			if code in codes:
				return {"success": False, "error_code": 8004, "error": "This user has been registered!"}

			path_code = os.path.join(PATH_IMG_AVATAR, code)
			if os.path.exists(path_code):
				shutil.rmtree(path_code)
			os.mkdir(path_code)

			uploaded_files = args['images']
			# print(len(uploaded_files))
			imgs = []
			for i, uploaded_file in enumerate(uploaded_files):
				in_memory_file = io.BytesIO()
				uploaded_file.save(in_memory_file)
				in_memory_file.seek(0)
				pil_img = Image.open(in_memory_file)
				img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
				cv2.imwrite(f'{path_code}/face_{i+1}.jpg', img)
				imgs.append(img)

			# #---------------------------face det-------------------------
			miss_det = []
			dets, miss_det = facedet.inference(imgs)
			if len(dets) == 0:
				return {"success": False, "error_code": 8001, "error": "Don't find any face"}
			print(dets)
			# #////////////////////////////////////////////////////////////

			#---------------------------face reg-------------------------
			# feature = facereg.get_feature_without_det(imgs)
			feature = facereg.get_feature(imgs, dets)
			feature = np.array([feature], dtype=np.float16)

			# db_ft = None
			db_ft = "db_1.npy"
			path_db = os.path.join(PATH_DB_FT, db_ft)
			if not os.path.exists(path_db):
				np.save(path_db, feature)
			else:
				# db_ft = f"db_1.npy"
				# path_db = os.path.join(PATH_DB_FT, db_ft)
				features = np.load(path_db).astype(np.float16)
				#--------add------
				# print(features.shape)
				# print(feature.shape)
				features = np.concatenate((features, feature), axis=0)
				#-----subtract------
				# features = np.delete(features, 1, axis=0)
				np.save(path_db, features)

			ur = User(code=code, name=name, birthday=birthday, avatar=f"{IMG_AVATAR}/{code}/face_1.jpg", feature_db=db_ft)
			ur.save()

			return {"success": True, "miss_face": f"Cannot find face in image {miss_det}"}
		except Exception as e:
			return {"success": False, "error_code": 8008, "error": str(e)}

upload_parser_deleteUser = api.parser()
upload_parser_deleteUser.add_argument("code", type=str, required=True)
@api.expect(upload_parser_deleteUser)
@api.route('/deleteUser')
class deleteUser(Resource):
	def post(self):
		try:
			args = upload_parser_deleteUser.parse_args()
			code = args["code"]
			path_code = os.path.join(PATH_IMG_AVATAR, code)

			codes = db.session.query(User.code).all()
			if len(codes) == 0:
				return {"success": False, "error_code": 8005, "error": "No users have been registered!"}
			codes = np.array(codes).squeeze(1)
			# print("---------------codes.shape: ", codes.shape)
			if code not in codes:
				return {"success": False, "error_code": 8006, "error": "This user has not been registered!"}

			db_ft = f"db_1.npy"
			path_db = os.path.join(PATH_DB_FT, db_ft)
			features = np.load(path_db).astype(np.float16)
			# print("--------------features.shape: ", features.shape)

			# delete user in db feature
			idx_usr = np.where(codes == code)[0][0]
			# print("-------------idx_usr: ", idx_usr)
			features = np.delete(features, idx_usr, axis=0)
			# print("--------------features.shape: ", features.shape)
			np.save(path_db, features)

			# delete user in mysql
			db.session.query(User).filter_by(code=code).delete()
			db.session.commit()

			# delete user image
			if os.path.exists(path_code):
				shutil.rmtree(path_code)

			return {"success": True}

		except Exception as e:
			return {"success": False, "error_code": 8008, "error": str(e)}

# upload_parser_deleteAllUser = api.parser()
# upload_parser_deleteAllUser.add_argument("code", type=str, required=True)
# @api.expect(upload_parser_deleteAllUser)
@api.route('/deleteAllUser')
class deleteAllUser(Resource):
	def get(self):
		try:
			# delete db feature
			db_ft = f"db_1.npy"
			path_db = os.path.join(PATH_DB_FT, db_ft)
			if os.path.exists(path_db):
				os.remove(path_db)

			# delete all user in mysql
			db.session.query(User).delete()
			db.session.commit()

			# delete all user image
			if os.path.exists(PATH_IMG_AVATAR):
				shutil.rmtree(PATH_IMG_AVATAR)
				os.mkdir(PATH_IMG_AVATAR)
			return {"success": True}

		except Exception as e:
			return {"success": False, "error_code": 8008, "error": str(e)}

upload_parser_getInformationUser = api.parser()
upload_parser_getInformationUser.add_argument("code", type=str, required=True)
@api.expect(upload_parser_getInformationUser)
@api.route('/getInformationUser')
class getInformationUser(Resource):
	def post(self):
		try:
			args = upload_parser_deleteUser.parse_args()
			code = args["code"]

			codes = db.session.query(User.code).all()
			if len(codes) == 0:
				return {"success": False, "error_code": 8005, "error": "No users have been registered!"}
			codes = np.array(codes).squeeze(1)
			if code not in codes:
				return {"success": False, "error_code": 8006, "error": "This user has not been registered!"}

			user = db.session.query(User).filter(User.code == code).all()[0]
			infor = {"code": user.code, "name": user.name, "birthday": user.birthday, "avatar": user.avatar}
			return {"success": True, "Information": infor}

		except Exception as e:
			return {"success": False, "error_code": 8008, "error": str(e)}

upload_parser_searchUser = api.parser()
upload_parser_searchUser.add_argument("image", location='files', type=FileStorage, required=True)
@api.expect(upload_parser_searchUser)
@api.route('/searchUser')
class searchUser(Resource):
	def post(self):
		try:
			args = upload_parser_searchUser.parse_args()
			uploaded_file = args['image']

			st_time = time.time()
			in_memory_file = io.BytesIO()
			uploaded_file.save(in_memory_file)
			in_memory_file.seek(0)
			pil_img = Image.open(in_memory_file)
			img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

			users = db.session.query(User).all()
			if len(users) == 0:
				return {"success": False, "error_code": 8000, "error": "Don't have any registered user"}

			db_ft = f"db_1.npy"
			path_db = os.path.join(PATH_DB_FT, db_ft)
			features = np.load(path_db).astype(np.float16)
			print("---------Time get db: ", time.time() - st_time)

			#---------------------------face det-------------------------
			dets, miss_det = facedet.inference([img])
			# facedet.render([img])
			if len(dets) == 0:
				return {"success": False, "error_code": 8001, "error": "Don't find any face"}
			#---------------spoofing--------------
			bboxes = dets[0]["loc"]
			biggestBox = None
			maxArea = 0
			for j, bboxe in enumerate(bboxes):
				x1, y1, x2, y2 = bboxe
				area = (x2-x1) * (y2-y1)
				if area > maxArea:
					maxArea = area
					biggestBox = bboxe
			x1, y1, x2, y2 = biggestBox
			w, h = (x2-x1, y2-y1)
			xyxy_new = np.array([x1-w/2, y1-h/2, x2+w/2, y2+h/2]).astype(int)
			xyxy_new[xyxy_new<0] = 0
			img_spoofing = img[xyxy_new[1]:xyxy_new[3], xyxy_new[0]:xyxy_new[2]].copy()
			# img_spoofing = img
			result = spoofingdet.inference([img_spoofing])[0]
			print("---------result_spoofing", result)
			if result[1] > 0.85:
				img_list = os.listdir("./image_test")
				cv2.imwrite(f"./image_test/{len(img_list)}.jpg", img_spoofing)
				return {"success": False, "error_code": 8002, "error": "Fake face image"}
			#//////////////////////////////////////
			#////////////////////////////////////////////////////////////
			print("---------Time detect: ", time.time() - st_time)
			# feature = facereg.get_feature_without_det([img])
			feature = facereg.get_feature([img], dets)
			feature = np.array(feature, dtype=np.float16)

			print("---------Time align: ", time.time() - st_time)
			similarity, idx_sorted = facereg.compare_face_1_n_n(feature, features)
			similarity_best = similarity[idx_sorted[0]]
			print("--------Time reg: ", time.time() - st_time)

			result = None
			print("---------similarity_best: ", similarity_best)
			print(len(users))
			# print(len(idx_sorted))
			if similarity_best > 0.70:
				result = users[idx_sorted[0]]
			print("---------Duration: ", time.time()-st_time)

			if result is None:
				return {"success": False, "error_code": 8003, "error": "Don't find any user"}
			print(result.code)
			print(result.name)

			return {"success": True, "Information": {"id": result.code, "name": result.name, "birthday": result.birthday, "avatar": result.avatar, "similarity": float(similarity_best)}}

		except Exception as e:
			return {"success": False, "error_code": 8008, "error": str(e)}

upload_parser_spoofing = api.parser()
upload_parser_spoofing.add_argument("image", location='files', type=FileStorage, required=True)
@api.expect(upload_parser_spoofing)
@api.route('/spoofingCheck')
class spoofingCheck(Resource):
	def post(self):
		try:
			args = upload_parser_searchUser.parse_args()
			uploaded_file = args['image']

			in_memory_file = io.BytesIO()
			uploaded_file.save(in_memory_file)
			in_memory_file.seek(0)
			pil_img = Image.open(in_memory_file)
			img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
			#---------------------------face det-------------------------
			dets, miss_det = facedet.inference([img])
			if len(dets) == 0:
				return {"success": False, "error_code": 8001, "error": "Don't find any face"}
			bboxes = dets[0]["loc"]
			biggestBox = None
			maxArea = 0
			for j, bboxe in enumerate(bboxes):
				x1, y1, x2, y2 = bboxe
				area = (x2-x1) * (y2-y1)
				if area > maxArea:
					maxArea = area
					biggestBox = bboxe
			x1, y1, x2, y2 = biggestBox
			w, h = (x2-x1, y2-y1)
			xyxy_new = np.array([x1-w/2, y1-h/2, x2+w/2, y2+h/2]).astype(int)
			xyxy_new[xyxy_new<0] = 0
			img = img[xyxy_new[1]:xyxy_new[3], xyxy_new[0]:xyxy_new[2]]
			# cv2.imwrite("sdvds.jpg", img)
			#/////////////////////////////////////////////////////////////
			result = spoofingdet.inference([img])[0]
			print("---------result_spoofing", result)
			if result[1] > 0.85:
				return {"success": False, "error_code": 8002, "error": "Fake face image"}
			return {"success": True}
		except Exception as e:
			return {"success": False, "error_code": 8008, "error": str(e)}

@api.route('/healthcheck')
class health_check(Resource):
	def get(self):
		return { 'success': True, 'message': "healthy" }

upload_parser_faceRegister2 = api.parser()
upload_parser_faceRegister2.add_argument("code", type=str, required=True)
upload_parser_faceRegister2.add_argument("name", type=str, required=True)
upload_parser_faceRegister2.add_argument("birthday", type=str, required=True)
upload_parser_faceRegister2.add_argument("images", location='files', type=FileStorage, required=True, action='append')
@api.expect(upload_parser_faceRegister2)
@api.route('/registerFacev2')
class registerFacev2(Resource):
	def post(self):
		try:
			args = upload_parser_faceRegister2.parse_args()
			code = args["code"]
			name = args["name"]
			birthday = args["birthday"]

			# codes = db.session.query(User2.code).all()
			# codes = list(chain(*codes))
			# if code in codes:
			# 	return {"success": False, "code": 8004, "error": "This user has been registered!"}
			path_code = os.path.join(PATH_IMG_AVATAR, code)
			if os.path.exists(path_code):
				shutil.rmtree(path_code)
			os.mkdir(path_code)

			uploaded_files = args['images']
			# print(len(uploaded_files))
			imgs = []
			for i, uploaded_file in enumerate(uploaded_files):
				in_memory_file = io.BytesIO()
				uploaded_file.save(in_memory_file)
				in_memory_file.seek(0)
				pil_img = Image.open(in_memory_file)
				img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
				# num_face = len(os.listdir(path_code))
				# cv2.imwrite(f'{path_code}/face_{num_face}.jpg', img)
				imgs.append(img)

			# #---------------------------face det-------------------------
			miss_det = []
			dets, miss_det = facedet.inference(imgs)
			if len(dets) == 0:
				return {"success": False, "error_code": 8001, "error": "Don't find any face"}
			# print(dets)
			# #////////////////////////////////////////////////////////////

			#---------------------------face reg-------------------------
			# feature = facereg.get_feature_without_det(imgs)
			feature = facereg.get_feature(imgs, dets)
			feature = np.array(feature, dtype=np.float16)

			# db_ft = None
			db_ft = "db_2.npy"
			path_db = os.path.join(PATH_DB_FT, db_ft)
			if not os.path.exists(path_db):
				np.save(path_db, feature)
			else:
				features = np.load(path_db).astype(np.float16)
				#--------add------
				# print(features.shape)
				# print(feature.shape)
				features = np.concatenate((features, feature), axis=0)
				#-----subtract------
				# features = np.delete(features, 1, axis=0)
				np.save(path_db, features)

			[imgs.pop(idx) for idx in reversed(sorted(miss_det))]
			for i, det in enumerate(dets):
				num_face = len(os.listdir(path_code))
				cv2.imwrite(f'{path_code}/face_{num_face}.jpg', imgs[i])
				cdl = CodeList(code=code, avatar=f"{IMG_AVATAR}/{code}/face_{num_face}.jpg")
				cdl.save()

			codes = db.session.query(User2.code).all()
			codes = list(chain(*codes))
			if code not in codes:
				ur = User2(code=code, name=name, birthday=birthday, avatar=f"{IMG_AVATAR}/{code}/face_0.jpg", feature_db=db_ft)
				ur.save()

			return {"success": True, "miss_face": f"Cannot find face in image {miss_det}"}

		except Exception as e:
			return {"success": False, "error_code": 8008, "error": str(e)}

@api.route('/deleteAllUserv2')
class deleteAllUserv2(Resource):
	def get(self):
		try:
			# delete db feature
			db_ft = f"db_2.npy"
			path_db = os.path.join(PATH_DB_FT, db_ft)
			if os.path.exists(path_db):
				os.remove(path_db)

			# delete all user in mysql
			db.session.query(User2).delete()
			db.session.commit()

			# delete all list code in mysql
			db.session.query(CodeList).delete()
			db.session.commit()

			# delete all user image
			if os.path.exists(PATH_IMG_AVATAR):
				shutil.rmtree(PATH_IMG_AVATAR)
				os.mkdir(PATH_IMG_AVATAR)
			return {"success": True}

		except Exception as e:
			return {"success": False, "error_code": 8008, "error": str(e)}

upload_parser_deleteUser2 = api.parser()
upload_parser_deleteUser2.add_argument("code", type=str, required=True)
@api.expect(upload_parser_deleteUser2)
@api.route('/deleteUserv2')
class deleteUserv2(Resource):
	def post(self):
		try:
			args = upload_parser_deleteUser.parse_args()
			code = args["code"]
			path_code = os.path.join(PATH_IMG_AVATAR, code)

			codes = db.session.query(User2.code).all()

			if len(codes) == 0:
				return {"success": False, "error_code": 8005, "error": "No users have been registered!"}
			codes = np.array(codes).squeeze(1)
			# print("---------------codes.shape: ", codes.shape)
			if code not in codes:
				return {"success": False, "error_code": 8006, "error": "This user has not been registered!"}

			code_list = db.session.query(CodeList.code).all()
			code_list = np.array(code_list).squeeze(1)
			idx_code = np.where(code_list == code)

			db_ft = f"db_2.npy"
			path_db = os.path.join(PATH_DB_FT, db_ft)
			features = np.load(path_db).astype(np.float16)
			print("--------------features.shape: ", features.shape)

			# # delete user in db feature
			# idx_usr = np.where(codes == code)[0][0]
			# # print("-------------idx_usr: ", idx_usr)
			features = np.delete(features, idx_code, axis=0)
			print("--------------features.shape: ", features.shape)
			np.save(path_db, features)

			# delete user in mysql
			db.session.query(User2).filter_by(code=code).delete()
			db.session.commit()

			# delete all list code in mysql
			db.session.query(CodeList).filter_by(code=code).delete()
			db.session.commit()

			# delete user image
			if os.path.exists(path_code):
				shutil.rmtree(path_code)

			return {"success": True}

		except Exception as e:
			return {"success": False, "error_code": 8008, "error": str(e)}

upload_parser_getInformationUser2 = api.parser()
upload_parser_getInformationUser2.add_argument("code", action='append', type=str)
@api.expect(upload_parser_getInformationUser2)
@api.route('/getInformationUserv2')
class getInformationUserv2(Resource):
	def post(self):
		try:
			args = upload_parser_getInformationUser2.parse_args()
			code_list = args["code"]
			print(code_list)

			codes = db.session.query(User2.code).all()
			if len(codes) == 0:
				return {"success": False, "error_code": 8005, "error": "No users have been registered!"}

			codes = np.array(codes).squeeze(1)
			infor_persons = {}
			if code_list is None:
				users = db.session.query(User2).all()
				print(users)
				for usr in users:
					infor_persons[usr.code] = {"id": usr.code, \
												"name": usr.name, \
												"birthday": usr.birthday, \
												"avatar": usr.avatar}
			else:
				for code in code_list:
					if code not in codes:
						infor_persons[code] = "No register"
						continue
					usr = db.session.query(User2).filter(User2.code == code).all()[0]
					infor_persons[code] = {"id": usr.code, \
												"name": usr.name, \
												"birthday": usr.birthday, \
												"avatar": usr.avatar}

			return {"success": True, "Information": infor_persons}

		except Exception as e:
			return {"success": False, "error_code": 8008, "error": str(e)}

upload_parser_searchUser2 = api.parser()
upload_parser_searchUser2.add_argument("image", location='files', type=FileStorage, required=True)
@api.expect(upload_parser_searchUser2)
@api.route('/searchUserv2')
class searchUserv2(Resource):
	def post(self):
		try:
			args = upload_parser_searchUser2.parse_args()
			uploaded_file = args['image']

			st_time = time.time()
			in_memory_file = io.BytesIO()
			uploaded_file.save(in_memory_file)
			in_memory_file.seek(0)
			pil_img = Image.open(in_memory_file)
			img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

			users = db.session.query(User2).all()
			if len(users) == 0:
				return {"success": False, "error_code": 8000, "error": "Don't have any registered user"}

			db_ft = f"db_2.npy"
			path_db = os.path.join(PATH_DB_FT, db_ft)
			features = np.load(path_db).astype(np.float16)
			print("---------Time get db: ", time.time() - st_time)

			code_list = db.session.query(CodeList.code).all()
			code_list = np.array(code_list).squeeze(1)
			codes, idx = np.unique(code_list, return_inverse=True)
			#---------------------------face det-------------------------
			dets, miss_det = facedet.inference([img])
			# facedet.render([img])
			if len(dets) == 0:
				return {"success": False, "error_code": 8001, "error": "Don't find any face"}
			#---------------spoofing--------------
			bboxes = dets[0]["loc"]
			biggestBox = None
			maxArea = 0
			for j, bboxe in enumerate(bboxes):
				x1, y1, x2, y2 = bboxe
				area = (x2-x1) * (y2-y1)
				if area > maxArea:
					maxArea = area
					biggestBox = bboxe
			x1, y1, x2, y2 = biggestBox
			w, h = (x2-x1, y2-y1)
			xyxy_new = np.array([x1-w/2, y1-h/2, x2+w/2, y2+h/2]).astype(int)
			xyxy_new[xyxy_new<0] = 0
			img_spoofing = img[xyxy_new[1]:xyxy_new[3], xyxy_new[0]:xyxy_new[2]].copy()
			# img_spoofing = img
			result = spoofingdet.inference([img_spoofing])[0]
			print("---------result_spoofing", result)
			if result[1] > 0.85:
				# img_list = os.listdir("./image_test")
				# cv2.imwrite(f"./image_test/{len(img_list)}.jpg", img_spoofing)
				return {"success": False, "error_code": 8002, "error": "Fake face image"}
			#//////////////////////////////////////
			#////////////////////////////////////////////////////////////
			print("---------Time detect: ", time.time() - st_time)
			# feature = facereg.get_feature_without_det([img])
			feature = facereg.get_feature([img], dets)
			feature = np.array(feature, dtype=np.float16)
			# print(feature.shape)
			# print(features.shape)

			print("---------Time align: ", time.time() - st_time)
			similarity, idx_sorted = facereg.compare_face_1_n_1(feature, features)
			# print(similarity)
			# print(idx_sorted)
			similarity_average = np.bincount(idx.flatten(), weights = similarity.flatten())/np.bincount(idx.flatten())	# calculate average of simililarity has the same code
			# print(similarity_average)
			rand = np.random.random(similarity_average.size)
			idx_sorted = np.lexsort((rand,similarity_average))[::-1] #sort random index by similarity_average
			# print(idx_sorted)
			similarity_best = similarity_average[idx_sorted[0]]
			print("--------Time reg: ", time.time() - st_time)

			result = None
			# print("---------similarity_best: ", similarity_best)
			# print(len(users))
			# # print(len(idx_sorted))
			if similarity_best > 0.70:
				code = codes[idx_sorted[0]]
				result = db.session.query(User2).filter(User2.code == code).all()[0]
			# print("---------Duration: ", time.time()-st_time)

			if result is None:
				return {"success": False, "error_code": 8003, "error": "Don't find any user"}
			# print(result.code)
			# print(result.name)

			return {"success": True, "Information": {"id": result.code, "name": result.name, "birthday": result.birthday, "avatar": result.avatar, "similarity": float(similarity_best)}}

		except Exception as e:
			return {"success": False, "error_code": 8008, "error": str(e)}

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=4444, debug=True, threaded=True)

# #gunicorn -w 4 controllers_face:app --threads 10 -b 0.0.0.0:4444
# """
# 8000: "Don't have any registered user"
# 8001: "Don't find any face"
# 8002: "Fake face image"
# 8003: "Don't find any user"
# 8004: "This user has been registered!"
# 8005: "No users have been registered!"
# 8006: "This user has not been registered!"
# 8008: error system
# """