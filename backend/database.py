from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = "sqlite:///./fraud_ledger.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class TransactionLedger(Base):
    __tablename__ = "transaction_ledger"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(64), index=True, nullable=False)
    txn_uuid = Column(String(64), unique=True, index=True, nullable=False)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    type = Column(String(32), nullable=False)
    amount = Column(Float, nullable=False)
    oldbalanceOrg = Column(Float, nullable=False)
    newbalanceOrig = Column(Float, nullable=False)
    oldbalanceDest = Column(Float, nullable=False, default=0.0)
    newbalanceDest = Column(Float, nullable=False, default=0.0)
    risk_score = Column(Float, nullable=False)
    decision = Column(String(16), nullable=False)
    heuristic_triggered = Column(String(128), nullable=True)

    model_votes = Column(String(256), nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
