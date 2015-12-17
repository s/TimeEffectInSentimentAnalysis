# -*- coding: utf-8 -*-


from sqlalchemy import Column
from sqlalchemy.types import *
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class TurkcellMergedTweet(Base):

    __tablename__ = 'TurkcellMergedTweets'
    id = Column(String(30), primary_key=True)
    text = Column(String(512))
    created_at = Column(TIMESTAMP())
    tweet_class = Column(Enum('positive', 'negative', 'neutral', name='tweet_class_type'))

    def __init__(self):
        pass
