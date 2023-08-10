# # function for converting data from original format to .csv format
def convert(imgf, labelf, outf, n):
    file = open(imgf, 'rb')
    output = open(outf, 'w')
    label = open(labelf, 'rb') 
    
    file.read(16)
    label.read(8)
    images = []
    
    for i in range(n):
        image = [ord(label.read(1))]
        for j in range(28*28):
            image.append(ord(file.read(1)))
        images.append(image)
        
    for image in images:
        output.write(','.join(str(pix) for pix in image) + '\n')
    file.close()
    output.close()
    label.close()
convert("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "mnist_train.csv", 60000)
convert("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
        "mnist_test.csv", 10000)

