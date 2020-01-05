from nltk.sentiment.vader import SentimentIntensityAnalyzer
from utils.preprocessor import preprocess_input
from utils.inference import apply_offsets
from keras.models import load_model
import matplotlib.pyplot as plt
from imutils import face_utils
from const import *
import numpy as np
from api import *
import requests
import dropbox
import time
import dlib
import glob
import cv2
import os

api = api()

analyzer = SentimentIntensityAnalyzer()


# get text file from dropox
def get_from_dropbox():
    TOKEN = "YOUR DROPBOX API TOKEN"
    dbx = dropbox.Dropbox(TOKEN)
    while True:
        res = dbx.files_list_folder(DROPBOX_PATH, recursive=True)
        for entry in res.entries:
            try:
                dbx.files_download_to_file("searchword.txt", DROPBOX_PATH + entry.name)  # dropboxからテキストファイルをダウンロード
                dbx.files_delete(DROPBOX_PATH + entry.name)  # もらったテキストファイルを削除
                with open("searchword.txt") as f:  # もらったテキストファイルを一行ずつ読み込んでいく
                    s = f.readlines()
                """
                1行目:調べるキーワード
                2行目:画像を含んたサーチをするかどうかを判別する数字
                """
                os.remove("searchword.txt")  # テキストファイル
                return s
            except:
                pass


# analyze text
def sentiment_scores(the_tweet):
    snt = analyzer.polarity_scores(the_tweet)  # テキスト解析
    return snt


# detect how many face there are
def faces_q_and_a(filename):
    # 顔を見つける一連の処理
    image = cv2.imread(filename)
    cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector2(rgb_image, 1)
    return faces, image


# analyze face expression
def image_processing(image_file):
    faces, image = faces_q_and_a(image_file)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ANGRY, DISGUST, FEAR, HAPPY, SAD, SURPRISE, NEUTRAL = [], [], [], [], [], [], []  # 感情の指標をそれぞれリストに
    emotion_list = [ANGRY, DISGUST, FEAR, HAPPY, SAD, SURPRISE, NEUTRAL]  # 感情の変数をリストに
    for face_coordinates in faces:  # 画像の中の感情を判定する
        x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), (20, 40))
        gray_face = gray_image[y1:y2, x1:x2]
        gray_face = cv2.resize(gray_face, (emotion_target_size))
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        print(emotion_classifier.predict(gray_face)[0])  # それぞれの感情の値
        for abc in range(7):
            emotion_list[abc].append(emotion_classifier.predict(gray_face)[0][abc])
            """
            emotion_listの0個目に、つまりANGRYにemotion_classifier.predict(gray_face)[0]の0個目を代入
            emotion_listの1個目に、つまりDISGUSTにemotion_classifier.predict(gray_face)[0]の1個目を代入
            ...
            """
    cv2.waitKey(0)
    topgun = len(faces) - 1  # 変数名がtopgunなのはtopgunを見ながらこのコードを見ていたから
    for e in range(1, topgun):
        for abc in range(7):  # それぞれの顔のそれぞれの感情を全て足していく（？）
            emotion_list[abc][e] = np.array(emotion_list[abc][e - 1] + np.array(emotion_list[abc][e]))
    # negative=================================
    negative = (np.array(ANGRY[topgun]) + np.array(DISGUST[topgun]) + np.array(FEAR[topgun]) + np.array(
        SAD[topgun])) / np.array([4])  # negative要素を全て足す
    # neutral==================================
    neutral = np.array(SURPRISE[topgun]) + np.array(NEUTRAL[topgun]) / np.array([2])  # neutral要素を全て足す
    # positive================================
    positive = np.array(HAPPY[topgun]) + np.array(SURPRISE[topgun]) / np.array([2])  # positive要素を全て足す

    magnification = 1 / (negative[0] + neutral[0] + positive[0])
    # negative, neutral, positiveを全て足しても合計は1とならないので、合計して1になるような倍率を算出する。

    nagative_value = negative[0] * magnification

    neutral_value = neutral[0] * magnification

    positive_value = positive[0] * magnification

    emotion_prediction = [nagative_value, neutral_value, positive_value]  # 合計して1になっている
    return emotion_prediction


# download image
def download_img(url, file_name):
    r = requests.get(url, stream=True)
    if r.status_code == 200:  # レスポンスを確認
        with open(file_name, 'wb') as f:
            f.write(r.content)


# (仮)the way to calculate text scores with emotion scores
def calculate(text, image):
    print(f"\n*******Faces were detected*********\ntext: {text}\nimage: {image}\n")
    return (np.array(text) + np.array(image)) / 2


def calculate_all(x):
    i = len(x)
    print(f"Tweet Count: {i}")
    for g in range(1, i):
        x[g] = np.array(x[g - 1]) + np.array(x[g])
    if len(x[i - 1]) == 0:
        api.update_with_media(status=f"There are enough tweet about {search_key_word}")
    else:
        pie_chart(x[i - 1])


def pie_chart(y):
    print(f"::::::::::value {y}::::::::::::::")
    label = ["negative", "neutral", "positive"]  # 要素の名前
    colors = ["skyblue", "springgreen", "orangered"]  # 色指定
    plt.figure()  # これないと二回目以降の円グラフ表示で1回目、2回目、...の結果も表示されてしまう。
    plt.title(search_key_word, fontsize=15)  # グラフのタイトル
    plt.pie(y, labels=label, counterclock=True, startangle=90, colors=colors, autopct="%1.1f%%",
            pctdistance=0.7)  # チャート作成
    plt.savefig("pi.jpg")  # チャートを保存
    api.update_with_media(status=f"I searched about {search_key_word}{measure}\ncount: {count}",
                          filename='pi.jpg')  # Twitterに投稿


def main(keyword, yes_or_no):
    c = 0
    n = {}
    global count
    count = 100
    for tweet in api.search(q=f"{keyword} exclude:retweets", count=count, lang='en'):  # qはキーワード excludeはリツイート避け
        try:
            print(tweet.text)
            text_score = sentiment_scores(tweet.text)
            del text_score['compound']  # 辞書のcompoundの要素を削除
            text_score = list(text_score.values())  # text_scoreの値だけを取得して、リスト化
            if yes_or_no == "1":  # including image processing
                try:  # ここをtryにしたのは下のurlでmediaが含まれない時にこのif文の処理を終了することなく、テキストだけで感情を判定できるようにするため
                    url = tweet.extended_entities['media'][0]['media_url']
                    filename = f"{PBL_IMAGE}\\image_{str(c)}.jpg"  # その画像の保存名とパス
                    download_img(url, filename)  # twitterのキーワードに関する写真を取得
                    if len(faces_q_and_a(filename)[0]) > 0:  # もし画像に顔が一つ以上検出されたら
                        image_score = image_processing(filename)  # その画像に含まれる顔の平均的な感情の値
                        n[c] = calculate(text_score, image_score)  # ツイートのテキストと上の値の平均的な値を算出
                    else:
                        n[c] = text_score  # 顔がなかったらテキストだけで感情を判定
                except:
                    """
                    import traceback
                    traceback.print_exc()
                    """
                    n[c] = text_score  # 画像がなかったらテキストだけで感情を判定
            else:  # テキストだけで判定
                n[c] = text_score
            print(n[c])
            c += 1
        except:
            """
            import traceback
            traceback.print_exc()
            """
            pass
    time.sleep(0.5)
    for file in glob.glob(f"{PBL_IMAGE}\\image_*.jpg"): os.remove(file)
    calculate_all(n)


if __name__ == '__main__':
    # prepreparing============================================
    detector2 = dlib.get_frontal_face_detector()
    emotion_classifier = load_model(EMOTION_MODEL)
    emotion_target_size = emotion_classifier.input_shape[1:3]
    # ========================================================
    while True:
        print("\nI'm waiting for your search word...\n")
        global search_key_word
        global measure
        search_key_word, b = get_from_dropbox()  # 左から順にキーワード、YESかNOかを表す数字
        if b == "1":
            procedure = f"I gonna search about {search_key_word} with text and image face information\n"
            measure = "Text and Image"
        else:
            procedure = f"I gonna search about {search_key_word} with text analysis\n"
            measure = "Only Text"
        print(f"{procedure}")
        try:
            main(search_key_word, b)

        except:
            import traceback

            traceback.print_exc()
            notification = "***エラーが発生したためこの処理は終了されました***"
            payload = {"value1": notification, "value2": traceback.format_exc(chain=True)}
            url = "https://maker.ifttt.com/trigger/error/with/key/orpDw7U1JiFj2NY_aHumYeBFNmUYiIWTQPwkukuxN28"
            response = requests.post(url, data=payload)
            print(notification)
