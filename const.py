import os
PBL_FILE_PATH = "YOUR PBL FOLDER PATH"  # ここにこのプログラムのファイルの場所を入れてください
PBL_IMAGE = os.path.join(PBL_FILE_PATH, "image")
# emotion
EMOTION_MODEL = os.path.join(PBL_FILE_PATH, "models", "emotion_model.hdf5")
SHAPE_PREDICTOR = os.path.join(PBL_FILE_PATH, "models", "shape_predictor_68_face_landmarks.dat")
#dropbox
DROPBOX_PATH = "/pbl/"  # Dropbox