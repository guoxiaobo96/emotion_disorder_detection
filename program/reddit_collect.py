import praw
reddit = praw.Reddit(client_id="qOCFmW7rYl8P2A", client_secret="-RxOwpvQts9Y7lxmkFk-p9lleZc",
                     password="19960906", user_agent="testscript by /u/gxb_96",
                     username="gxb_96")
for submission in reddit.subreddit("learnpython").hot(limit=10):
    print(submission.title)