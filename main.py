from telegram.ext import Updater
from telegram.ext import Updater, CommandHandler ,MessageHandler
from telegram.ext import MessageHandler, Filters
import cv2
from io import BytesIO
import numpy as np
import tensorflow as tf

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar100.load_data()
x_train,x_test=x_train/255,x_test/255


items = ["beaver", "dolphin", "otter", "seal", "whale", "aquarium fish", "flatfish", "ray", "shark", "trout", "orchids", "poppies", "roses", "sunflowers", "tulips", "bottles", "bowls", "cans", "cups", "plates", "apples", "mushrooms", "oranges", "pears", "sweet peppers", "clock", "computer keyboard", "lamp", "telephone", "television", "bed", "chair", "couch", "table", "wardrobe", "bee", "beetle", "butterfly", "caterpillar", "cockroach", "bear", "leopard", "lion", "tiger", "wolf", "bridge", "castle", "house", "road", "skyscraper", "cloud", "forest", "mountain", "plain", "sea", "camel", "cattle", "chimpanzee", "elephant", "kangaroo", "fox", "porcupine", "possum", "raccoon", "skunk", "crab", "lobster", "snail", "spider", "worm", "baby", "boy", "girl", "man", "woman", "crocodile", "dinosaur", "lizard", "snake", "turtle", "hamster", "mouse", "rabbit", "shrew", "squirrel", "maple", "oak", "palm", "pine", "willow", "bicycle", "bus", "motorcycle", "pickup truck", "train", "lawn-mower", "rocket", "streetcar", "tank", "tractor"]

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), padding = "same", activation='relu', input_shape = (32,32,3)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu"))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation = "relu"))
model.add(tf.keras.layers.Dense(10, activation = "softmax"))


def start(update,context):

    update.message.reply_text('''
    Hello !
    Welcome to TFTimageBOT!
    I'm a bot that searches for images with images provided by you 
    Use /help for commands
    ''')

def help(update,context):
    update.message.reply_text('''
    Here are the commands:
    /start-> to begin
    /help-> to list commands
    /train-> training the bot 
    
    No need to give any command for the image output,
    just upload the image 
    ''')



def train(update,context):
    update.message.reply_text("Model is being trained...")
    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
    model.fit(x_train, y_train, epochs = 5, validation_data = (x_test, y_test))
    model.save("cifar_classifier.model")
    update.message.reply_text("Training completed! Yo can now send a picture!")


def photohandler(update, context):
    file=context.bot.get_file(update.message.photo[-1].file_id)
    f=BytesIO(file.download_as_bytearray())
    file_bytes=np.asarray(bytearray(f.read()),dtype=np.uint8)

    img=cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img=cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

    prediction=model.predict(np.array([img/255]))
    update.message.reply_text(f"{items[np.argmax(prediction)]}")

def wordshandler(update, context):
    update.message.reply_text('''
    Please upload an image once you're done training!!!
    ''')
  

def main():
    TOKEN='5470734389:AAF6s8L3f-0ha64lfvbpdNH_2mnmqa_0g7I'
    updater=Updater(TOKEN,use_context=True)
    disp=updater.dispatcher

    disp.add_handler(CommandHandler("start",start))
    disp.add_handler(CommandHandler("help",help))
    disp.add_handler(CommandHandler("train",train))

    disp.add_handler(MessageHandler(Filters.text,wordshandler))
    disp.add_handler(MessageHandler(Filters.photo,photohandler))

    updater.start_polling()
    updater.idle()
    

if __name__=='__main__':
    main()