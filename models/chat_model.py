from db_init import db
from sqlalchemy import func

class ChatModel(db.Model):
    __table_args__ = {'schema': 'watch_tm'}

    __tablename__ = "chat"

    id = db.Column(db.Integer, primary_key=True, index=True)
    name = db.Column(db.String(100))
    reg_date = db.Column(db.Date)
    reg_id = db.Column(db.String(100))
    mod_date = db.Column(db.Date)
    mod_id = db.Column(db.String(100))
    sort = db.Column(db.Integer)

    def __init__(self, id, name, reg_date, reg_id, mod_date, mod_id, sort):
        self.id = id
        self.name = name
        self.reg_date = reg_date
        self.reg_id = reg_id
        self.mod_date = mod_date
        self.mod_id = mod_id
        self.sort = sort

    @classmethod
    def save(cls, chat):
        db.session.merge(chat)
        db.session.commit()
        db.session.close()

    @classmethod
    def max_sort(cls):
        result = db.session.query(func.coalesce(func.max(cls.sort), 0)).scalar()
        return result+1
    
    @classmethod
    def find_by_id(cls, id):
        chat= db.session.query(cls).filter_by(id=id).first()
        return chat
    
    @classmethod
    def delete_by_id(cls, id):
        try:
            db.session.query(cls).filter_by(id=id).delete(synchronize_session=False)
        except:
            db.session.rollback()
        finally:
            db.session.commit()
            db.session.close()

    @classmethod
    def update_by_id(cls, id, values):
        db.session.query(cls).filter_by(id=id).update(values)
        db.session.commit()
        db.session.close()

    def __str__(self):
        return "[" + str(self.__class__) + "]: " + str(self.__dict__)