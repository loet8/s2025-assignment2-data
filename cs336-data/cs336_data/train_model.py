import fasttext

model = fasttext.train_supervised(
    input="quality_train.txt",  
    epoch=15,                   
    lr=0.5,                     
    wordNgrams=2,               
    bucket=100000,              
    thread=4,                   
    minCount=3,                 
    verbose=2                   
)

model.save_model("quality_classifier.bin")
print("Model trained and saved as quality_classifier.bin")
