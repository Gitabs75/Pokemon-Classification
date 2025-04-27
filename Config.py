#Config.py
img_size=(128, 128)  #Size of IMage
btch_size=64  #Training Batch size
epchs=80  #Training epochs
l=1e-3   #LearningRate for AdamW optimizer
wd=1e-3  #weightdecay
t_size=0.1  
pbatch=32  #Batch size of predict.py
n_cls=18  #No of classes
tst_sze=0.1765   #% of total data used for testing 0.1765 * 0.85 ≈ 0.15 total for val

BSE = "/mnt/c/Users/Abhishek Menon/Desktop" #Base directory
IMG = os.path.join(BASE_DIR, "Data/Images") #You will find dataset here
MDL = os.path.join(BASE_DIR, "Checkpoints/model.pth") #Modelweights file
TRN_LOG = "resnet_training_log.csv" #Training log with train and validation loss respectively
otp_flder="splits"

TRN = os.path.join(BASE_DIR, otp_flder, "train.csv") #The three following files contain imagename and label of the train, validation and test set repsctiveley
VAL = os.path.join(BASE_DIR, otp_flder, "val.csv")
TST = os.path.join(BASE_DIR, otp_flder, "test.csv")
lbl_csv="pokemon_singlelabel.csv" #Contains all filenames and labels 

R_TRN =True #Boolean value runs training if true
R_TST =True #Boolean value does testing if true

CLS = [
    'Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting',
    'Fire', 'Flying', 'Ghost', 'Grass', 'Ground', 'Ice',
    'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water'
]
