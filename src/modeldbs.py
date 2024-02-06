from app_brief_s3_test import app, db

class User(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	code = db.Column(db.String(255), nullable=False, unique=True)
	name = db.Column(db.String(255), nullable=False)
	birthday = db.Column(db.String(255), nullable=False)
	avatar = db.Column(db.String(255), nullable=False)
	feature_db = db.Column(db.String(255), nullable=False)

	def save(self):
		db.session.add(self)
		db.session.commit()

# Setup database
with app.app_context():
    db.create_all()