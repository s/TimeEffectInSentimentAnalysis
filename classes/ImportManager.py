# -*- coding: utf-8 -*-
from DBManager import DBManager
from TwitterManager import TwitterManager
from PreprocessManager import PreprocessManager

from helpers.GeneralHelpers import GeneralHelpers

from models.TTNetTweet import TTNetTweet

class ImportManager:

    """
        This class imports tweets from txt files to MySQL database
    """

    __file_path = None
    __components_in_a_line = None

    def __init__(self, file_path_to_import):
        """
        :param file_path_to_import: String a txt file path containing tweet ids
        :return: ImportManager instance
        """

        self.__db_manager = DBManager()
        self.__helper = GeneralHelpers()
        self.__preprocess_manager = PreprocessManager()
        self.__file_path = file_path_to_import
        self.__tweets_classes_dictionary = {}

        # magic numbers
        self.__components_in_a_line = 2
        self.__max_num_of_tweets_at_once = 100

    def run(self):
        """
        Runs all necessary methods to import tweets for a year
        :return:
        """

        # getting tweets with their classes
        tweets_with_classes = self._parse_tweets_from_file()
        self.__tweets_with_classes = tweets_with_classes

        # finding duplicates
        unique_tweets, duplicate_tweets = self._find_duplicates(tweets_with_classes)

        print("Found "+str(len(duplicate_tweets))+" duplicate tweets.")
        self.__helper.pretty_print_list(duplicate_tweets, "Duplicate tweets:")
        print("Continuing with unique ones.")

        # getting tweet ids from [tweet_id, class]
        unique_tweets_ids = self._get_tweets_ids(unique_tweets)

        # retrieving tweets from Twitter
        all_tweet_information = self._retrieve_tweets_from_twitter(unique_tweets_ids)

        # some tweets may not be found on Twitter
        not_found_tweets_on_twitter = self._find_not_found_tweets_on_twitter(all_tweet_information)

        # creating db model objects
        all_tweet_objects = self._create_tweet_objects(all_tweet_information)

        # insert to database
        success_count, not_imported_tweets = self.__db_manager.insert_tweet_objects(all_tweet_objects)

        print("\n")
        print('-'*10)
        print('Total Math:')
        print('Unique tweets:'+str(len(unique_tweets)))
        print('Tweets not found:'+str(len(not_found_tweets_on_twitter)))
        print('Tweets not inserted:'+str(len(not_imported_tweets)))
        print('Tweets OK:'+str(success_count))
        print(str(len(unique_tweets))+"=="+str(len(not_found_tweets_on_twitter)+len(not_imported_tweets)+success_count))

    def _parse_tweets_from_file(self):
        """
        Parses tweet ids and classes from txt file
        :return: list, holds [[124214124, positive],...]
        """

        characters_to_remove = ["'", '"', '\n', ' ']

        with open(self.__file_path, 'r') as tweets_ids_file:
            tweets_with_classes = []
            self.tweets_classes_dictionary = {}

            # Iterating over lines in txt file
            for line in tweets_ids_file:
                line_components = line.split(",")

                # if there are two components in a line. E.g. "121412412412", "positive"
                if self.__components_in_a_line == len(line_components):

                    # iterating over components
                    for index, component in enumerate(line_components):

                        # removing unnecessary characters
                        line_components[index] = self.__preprocess_manager.remove_characters_in_string(component,
                                                                                                       characters_to_remove)

                    tweets_with_classes.append(line_components)
                    self.__tweets_classes_dictionary.update({line_components[0]:line_components[1]})

            return tweets_with_classes

    def _find_duplicates(self, tweets_with_classes):
        """
        Finds duplicate tweets
        :param tweets_with_classes: List a list of tweet ids and their sentiment classes.
        :return: unique tweets, duplicate tweets
        """

        unique_tweets = []
        seen_tweets_ids = []
        duplicate_tweet_ids = []

        # Iterating over tweets with their classes. E.g [[214124124124, positive], [124124124124, negative]...]
        for tweet_block in tweets_with_classes:

            # First element is the tweet id
            tweet_id = tweet_block[0]

            # If it isn't seen before
            if not tweet_id in seen_tweets_ids:
                seen_tweets_ids.append(tweet_id)
                unique_tweets.append(tweet_block)

            else:
                duplicate_tweet_ids.append(tweet_id)

        return unique_tweets, duplicate_tweet_ids

    def _retrieve_tweets_from_twitter(self, tweet_ids):
        """
        Retrieves tweet information from Twitter
        :param unique_tweets_with_classes: List, tweets and
        :return:
        """
        tweets_results = []
        twitter_manager = TwitterManager()

        chunks_of_tweets_ids = self.__helper.get_chunks_of_list(tweet_ids, self.__max_num_of_tweets_at_once)

        for chunk in chunks_of_tweets_ids:
            print("Searching for "+str(len(chunk))+" tweets.")
            result = twitter_manager.lookup(chunk)
            print("Found "+str(len(result))+" tweets.")
            tweets_results+=result

        return tweets_results

    def _get_tweets_ids(self, tweets_with_classes):
        """
        Extracts tweet ids from tweets with classes
        :param tweets_with_classes: List, tweet_with classes
        :return: extracted ids
        """
        ids = [x[0] for x in tweets_with_classes]
        return ids

    def _create_tweet_objects(self, all_tweets):
        """
        Creates a list which holds db model objects.
        :param tweet_info: List, tweets
        :return: List, tweet objects
        """

        all_tweet_objects = []

        for tweet in all_tweets:
            tweet_object = self.__db_manager.get_new_model_instance()
            tweet_object.id = tweet.id_str
            tweet_object.created_at = tweet.created_at
            tweet_object.lang = tweet.lang
            tweet_object.source = tweet.source
            tweet_object.user_id = tweet.user.id_str

            tweet_object.text = self.__preprocess_manager.clean_emojis_and_smileys(tweet.text).encode('utf-8')
            tweet_object.tweet_class = self._get_sentiment_class_of_tweet(tweet.id_str)

            all_tweet_objects.append(tweet_object)

        return all_tweet_objects

    def _get_sentiment_class_of_tweet(self, tweet_id):
        """
        Returns a sentiment class for given tweet id
        :param tweet_id: string
        :return: tweet class
        """
        return self.__tweets_classes_dictionary[tweet_id]

    def _find_not_found_tweets_on_twitter(self, twitter_response):
        """
        Finds not found tweets on Twitter API
        :param twitter_response: list, Twitter response
        :return: list, not found tweets
        """
        tweets_ids = self._get_tweets_ids(self.__tweets_with_classes)
        response_ids = [a_tweet_response.id_str for a_tweet_response in twitter_response]
        return list(set(tweets_ids) - set(response_ids))