# -*- coding: utf-8 -*-
from config import *

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.expression import extract

from models.TTNetTweet import TTNetTweet
from models.TurkcellTweet import TurkcellTweet


class DBManager:
    """
    This class manages database basic operations
    """

    def __init__(self):
        """
        Constructor method
        :return:
        """
        engine = create_engine('mysql+pymysql://root:@localhost/Thesis?charset=utf8')
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.engine = engine

    def insert_tweet_objects(self, tweet_objects):
        """
        Inserts tweet objects to database
        :param tweet_objects: list, objects of tweets
        :return: success_count, not_imported_tweets
        """
        not_imported_tweets = []
        successful_tweet_count = 0

        for tweet_object in tweet_objects:
            try:
                self.session.add(tweet_object)
                self.session.commit()
                successful_tweet_count += 1

            except Exception, e:
                not_imported_tweets.append(tweet_object)
                self.session.rollback()

        return successful_tweet_count, not_imported_tweets

    def get_tweets_for_year(self, year):
        """
        Returns tweet created at given year
        :param year: string, year
        :return: list, tweets matching criteria
        """

        model = self.get_new_model()
        if not 'ALL' == year:
            tweets = self.session.query(model).filter(
                year == extract('year', model.created_at)).order_by(
                model.created_at).all()
        else:
            tweets = self.session.query(model).order_by(model.created_at).all()

        print("Retrieved " + str(len(tweets)) + " rows from database for year:" + str(year) + ".")
        return tweets

    def get_new_model_instance(self):
        """
        Returns an instance of selected model in config.py
        :return: db model instancel
        """
        if "TTNet" == MODEL_NAME:
            return TTNetTweet()
        elif "Turkcell" == MODEL_NAME:
            return TurkcellTweet()

    def get_new_model(self):
        """
        Returns a model of selected model
        :return: db model itself
        """
        if "TTNet" == MODEL_NAME:
            return TTNetTweet
        elif "Turkcell" == MODEL_NAME:
            return TurkcellTweet

    def get_model_table_name(self):
        """
        Returns table name of model
        :return: string
        """
        return MODEL_NAME + 'Tweets'

    def get_years_tweets_counts(self, year):
        """
        Returns tweet count for given year
        :param year: string
        :return: int
        """
        model = self.get_new_model()
        tweets_count = self.session.query(model).filter(year == extract('year', model.created_at)).count()
        return tweets_count

    def get_months_lengths(self):
        """
        Returns monthly tweet counts of a given model
        :return: list
        """
        table_name = self.get_model_table_name()
        model_month_counts = {}
        q = """
            select
                count(*),
                month(created_at),
                year(created_at)
            from """ + table_name + """
            where
                created_at >= '2013-01-01'
            group by
                year(created_at),
                month(created_at)
            """
        month_lengths = self.engine.execute(q)

        for count, month, year in month_lengths:
            if not year in model_month_counts:
                model_month_counts[year] = []
            model_month_counts[year].append(count)

        return model_month_counts