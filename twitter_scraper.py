import tweepy

# Authorization keys
api_Key = "xiWtVeDNABUexhfiEJA2t2X69"                     # your api key
api_Secret = "AA0GwFO2UjI15vTI36SFV1Twgmps9iqqB60fxrYpKi3ziNzWVT"                  # your api Secret
access_token = "2613071485-9QeM7wuIT6q7EkpaGx1UPnxfxeEpUfP6bhaWmTn"
access_token_secret = "syoWUH0vwyyORUTSdy647XFd3Iqo4MOBjQ3AXMU3JifSu"


def get_tweets(hashtag, language, resultType, n):
    # Authorization to api key and api secret
    auth = tweepy.OAuthHandler(api_Key, api_Secret)

    # Access to user's access token and access token secret
    auth.set_access_token(access_token, access_token_secret)

    # Calling api
    api = tweepy.API(auth)
    tweets = tweepy.Cursor(api.search, q=hashtag, result_type=resultType, lang=language, tweet_mode='extended').items(n)

    tweets_list = [tweet.full_text for tweet in tweets]
    return tweets_list

