# -*- coding: utf-8 -*-

#################################################
# TurkcellTweet.py  		
# September 20
# TweetRetriever
#################################################

from sqlalchemy import Column
from sqlalchemy.types import *
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TurkcellTweet(Base):

	__tablename__ = 'TurkcellTweets'
	id = Column(String(30), primary_key=True)
	text =  Column(String(512))
	created_at = Column(TIMESTAMP())
	lang = Column(String(5))
	source = Column(String(128))
	user_id = Column(String(30))
	tweet_class = Column(Enum('positive', 'negative', 'neutral', name='tweet_class_type'))

	def __init__(self):
		pass