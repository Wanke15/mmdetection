import os


with open("./train_anno.txt", 'w') as f:
    for img in os.listdir('./train'):
        f.write(img.split('.')[0] + '\n')

with open("./test_anno.txt", 'w') as f:
    for img in os.listdir('./test'):
        f.write(img.split('.')[0] + '\n')
