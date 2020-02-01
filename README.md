# Overview

***This program tweets a pie chart that shows what the user thinks about the keyword by inputting the keyword you wanna search into the shortcuts app*** (provided by Apple).

After a text file containing a keyword is posted to a specific folder in Dropbox, the PC starts collecting the tweet data by downloading it.

You can select whether you wanna search with only tweet text or text + image of the tweet.

# Preparing

Prepare on the assumption that you already have TWITTER & DROPBOX API Key.

1. Install packages you need to do to run the program.

2. Download shortcuts recipe [GLOBAL PBL](https://www.icloud.com/shortcuts/4ff0c87bbce84a2688ce2a51601537c4) on your iPhone or iPad series.

3. Input your twitter API keys to api.py.

4. Input your DROPBOX API Key to main.py.

5. Input "PBL-Everyone-" folder location on your PC to PBL_FILE_PATH of const.py.

# Procedure

1. Run `python main.py`

2. Run the shortcut **GLOBAL PBL** and input search key word and answer yes or no question.

3. Keep watching over it...(Just a few second)

4. Check your tweet and you can know what you wanna know.

# How to work
![](https://github.com/yusuke-1105/Search-on-Twitter/blob/master/calculation.jpg)
  
This program outputs each value of ***Positive**, **Neutral** and **Negative***.  
But the image processing, which is a part of this program, returns the values of ***Angry**, **Disgust**, **Fear**, **Sad**, **Surprise**, **Neutral** and **Happy***.  
  
So There is a need for converging them to those values.   
  
For that reason, the above processing is performed.  
  
However, this is not an academic calculation method.  
  
***If you know the right calculation way, please tell me.***

# License

Copyright (c) 2020 Yusuke Sakabe  
Released under the [MIT license](https://github.com/yusuke-1105/Search-on-Twitter/blob/master/LICENSE)

# Credit

Provided and inspirated by  
https://github.com/oarriaga/face_classification  
https://github.com/petercunha/Emotion  
https://github.com/dropbox/dropbox-sdk-python  
