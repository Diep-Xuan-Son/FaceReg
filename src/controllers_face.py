from app_brief_s3_test import *
from modeldbs import User

upload_parser_faceRegister = api.parser()
upload_parser_faceRegister.add_argument("code", type=str, required=True)
upload_parser_faceRegister.add_argument("name", type=str, required=True)
upload_parser_faceRegister.add_argument("birthday", type=str, required=True)
upload_parser_faceRegister.add_argument("images", location='files', type=FileStorage, required=True, action='append')
@api.expect(upload_parser_faceRegister)
@api.route('/registerFace')
class registerFace(Resource):
	def post(self):
		args = upload_parser_faceRegister.parse_args()
		code = args["code"]
		name = args["name"]
		birthday = args["birthday"]

		codes = db.session.query(User.code).all()
		codes = list(chain(*codes))
		print(codes)
		if code in codes:
			return {"success": False, "error": "This user has been registered!"}

		path_code = os.path.join(PATH_IMG_AVATAR, code)
		if os.path.exists(path_code):
			shutil.rmtree(path_code)
		os.mkdir(path_code)

		uploaded_files = args['images']
		print(len(uploaded_files))
		imgs = []
		for i, uploaded_file in enumerate(uploaded_files):
			in_memory_file = io.BytesIO()
			uploaded_file.save(in_memory_file)
			in_memory_file.seek(0)
			pil_img = Image.open(in_memory_file)
			img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
			# cv2.imwrite(f'{path_code}/face_{i+1}.jpg', img)
			imgs.append(img)

		# #---------------------------face det-------------------------
		miss_det = []
		dets, miss_det = facedet.inference(imgs)
		if len(dets) == 0:
			return {"success": False, "error": "Don't find any face"}
		# print(dets)
		# #////////////////////////////////////////////////////////////

		#---------------------------face reg-------------------------
		# feature = facereg.get_feature_without_det(imgs)
		feature = facereg.get_feature(imgs, dets)
		feature = np.array([feature], dtype=np.float16)

		db_ft = None
		if len(os.listdir(PATH_DB_FT)) == 0:
			db_ft = "db_1.npy"
			path_db = os.path.join(PATH_DB_FT, db_ft)
			np.save(path_db, feature)
		else:
			db_ft = f"db_{len(os.listdir(PATH_DB_FT))}.npy"
			path_db = os.path.join(PATH_DB_FT, db_ft)
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
			codes = np.array(codes).squeeze(1)
			# print("---------------codes.shape: ", codes.shape)
			if code not in codes:
				return {"success": False, "error": "This user has not been registered!"}

			db_ft = f"db_{len(os.listdir(PATH_DB_FT))}.npy"
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
			return {"success": False, "error": str(e)}

# upload_parser_deleteAllUser = api.parser()
# upload_parser_deleteAllUser.add_argument("code", type=str, required=True)
# @api.expect(upload_parser_deleteAllUser)
@api.route('/deleteAllUser')
class deleteAllUser(Resource):
	def get(self):
		try:
			# delete db feature
			db_ft = f"db_{len(os.listdir(PATH_DB_FT))}.npy"
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
			return {"success": False, "error": str(e)}

upload_parser_searchUser = api.parser()
upload_parser_searchUser.add_argument("image", location='files', type=FileStorage, required=True)
@api.expect(upload_parser_searchUser)
@api.route('/searchUser')
class searchUser(Resource):
	def post(self):
		# try:
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
			return {"success": False, "error": "Don't have any registered user"}

		db_ft = f"db_{len(os.listdir(PATH_DB_FT))}.npy"
		path_db = os.path.join(PATH_DB_FT, db_ft)
		features = np.load(path_db).astype(np.float16)
		print("---------Time get db: ", time.time() - st_time)

		#---------------------------face det-------------------------
		# facedet.render([img])
		dets, miss_det = facedet.inference([img])
		if len(dets) == 0:
			return {"success": False, "error": "Don't find any face"}
		# print(dets)
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
		print(len(idx_sorted))
		if similarity_best > 0.70:
			result = users[idx_sorted[0]]
		print("---------Duration: ", time.time()-st_time)

		if result is None:
			return {"success": False, "error": "Don't find any user"}

		return {"success": True, "Information": {"code": result.code, "name": result.name, "birthday": result.birthday, "avatar": result.avatar, "similarity": float(similarity_best)}}

		# except Exception as e:
		# 	return {"success": False, "error": str(e)}

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=4444, debug=True, threaded=True)

#gunicorn -w 4 controllers_face:app --threads 10 -b 0.0.0.0:4444