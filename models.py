# models.py
from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
from sqlalchemy import ForeignKey


Base = declarative_base()

# -------------------- USER MODEL ---------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)         # <-- REQUIRED
    password = Column(String)                   # <-- REQUIRED
    created_at = Column(DateTime, default=datetime.utcnow)

# -------------------- HISTORY MODEL ---------------------
class SessionHistory(Base):
    __tablename__ = "session_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    disease = Column(String)
    symptoms = Column(String)
    timestamp = Column(DateTime)
    report_file = Column(String, nullable=True)   # ⭐ ADD THIS COLUMN



engine = create_engine("sqlite:///app.db")
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(engine)
