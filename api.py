import tweepy


def api():
    CONSUMER_KEY = 'YOURS'
    CONSUMER_SECRET = 'YOURS'
    ACCESS_TOKEN = 'YOURS'
    ACCESS_SECRET = 'YOURS'

    auth1 = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth1.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    api1 = tweepy.API(auth1, wait_on_rate_limit=True)
    return api1
