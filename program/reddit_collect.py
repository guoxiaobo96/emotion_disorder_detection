import praw
import json
import re
reddit_isntance_args = {"client_id": "qOCFmW7rYl8P2A", "client_secret": "-RxOwpvQts9Y7lxmkFk-p9lleZc",
                        "password": "19960906", "user_agent": "testscript by /u/gxb_96",
                        "username": "gxb_96"}

bipolar_subreddit_list = ["bipolar", "bipolar2",
                          "BipolarReddit", "BipolarSOs", "bipolarart", "mentalhealth"]


class GetFromReddit(object):
    def __init__(self, client_id, client_secret, password, user_agent, username):
        super().__init__()
        self._reddit_instance = praw.Reddit(client_id=client_id, client_secret=client_secret,
                                            password=password, user_agent=user_agent,
                                            username=username)
        self.comments = None


class GetSubredditComments(GetFromReddit):
    def __init__(self, client_id, client_secret, password, user_agent, username, subreddit_name):
        super().__init__(client_id, client_secret, password, user_agent, username)
        self._subreddit_name = subreddit_name
        self._subreddit = self._reddit_instance.subreddit(self._subreddit_name)

    def get_comments(self, comment_number, type='comment'):
        if type == 'comment':
            self.comments = self._subreddit.comments(limit=comment_number)
        elif type == 'hot':
            self.comments = self._subreddit.hot(limit=comment_number)
        comment_list = []
        for comment in self.comments:
            comment_list.append(comment)
        return comment_list


class GetUserComments(GetFromReddit):
    def __init__(self, client_id, client_secret, password, user_agent, username, reddit_username):
        super().__init__(client_id, client_secret, password, user_agent, username)
        self._reddit_username = reddit_username
        self._user = self._reddit_instance.redditor(self._reddit_username)
        self._comment_list = list()

    def get_comments(self, comment_number=1000, subreddit_list=[]):
        for comment in self._user.comments.new(limit=comment_number):
            comment_id = comment.id
            text = comment.body.replace('\n', '')
            subreddit_info = re.compile(r'r/[^ ]+')
            text = re.sub(subreddit_info, '', text)
            link_id = comment.link_id
            if not subreddit_list:
                self._comment_list.append({comment_id:{'text': text, 'link_id': link_id}})
            else:
                if comment.subreddit.display_name in subreddit_list:
                    self._comment_list.append({comment_id:{'text': text, 'link_id': link_id}})
        return self._comment_list

def get_bipolar_data():
    user_list = set()
    with open('data/bipolar/user_list', mode='r', encoding='utf8') as file:
        for line in file.readlines():
            user_list.add(line.split(' [info] ')[0])
    get_comments(user_list, bipolar_subreddit_list)

def get_comments(checked_user_list, subreddit_list):
    for subreddit_name in subreddit_list:
        get_reddit_comments = GetSubredditComments(
            **reddit_isntance_args, subreddit_name=subreddit_name)
        comments_list = get_reddit_comments.get_comments(1000)
        user_list = set()
        for comment in comments_list:
            if comment.author is not None and comment.author.name not in checked_user_list:
                user_list.add(comment.author.name)
        for user in user_list:
            get_user_comments = GetUserComments(
                **reddit_isntance_args, reddit_username=user)
            comment_list = get_user_comments.get_comments(comment_number=1000)
            if _identify_bipolar(user, comment_list):
                with open ('data/bipolar/'+user, mode='w', encoding='utf8') as fp:
                    for comment in comment_list:
                        comment = json.dumps(comment)
                        fp.write(comment+'\n')


def _identify_bipolar(username, comment_list):
    identify_list = ["I am diagnosed with bipolar", "i am diagnosed with bipolar",
                     "I'm diagnosed with bipolar", "i'm diagnosed with bipolar", "I have been diagnosed with bipolar",
                     "I was diagnosed with bipolar", "I've been diagnosed with bipolar"]

    for comment in comment_list:
        for key, value in comment.items():
            text = value['text']
            for sentence in identify_list:
                if sentence in text:
                    with open('data/bipolar/user_list', mode='a',encoding='utf8') as file:
                        file.write(username+' [info] '+text+'\n')
                    return True
    return False


if __name__ == '__main__':
    get_bipolar_data()
