import tweepy
from config import *

class TwitterManager:

    def __init__(self):
        """
        Creates a Twitter client with credentials
        :return: a TwitterManager instance
        """
        auth = tweepy.OAuthHandler(TWITTER_CONSUMER_TOKEN, TWITTER_CONSUMER_SECRET)
        auth.set_access_token(TWITTER_ACCESS_TOKEN_KEY, TWITTER_ACCESS_TOKEN_SECRET)
        self.api = tweepy.API(auth)

    def lookup(self, tweets_ids):
        """
        Returns tweets' detailed informations for a given tweets id list
        :param tweets_ids: List
        :return: Tweet objects, not found tweets
        """

        results = self.api.statuses_lookup(tweets_ids, trim_user=True)
        return results