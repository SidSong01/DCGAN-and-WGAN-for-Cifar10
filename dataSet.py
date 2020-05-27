import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pickle

def ImgShow(IMG,index,nums):

    fig, axes = plt.subplots(figsize=( nums+2,len(index)+2), nrows=len(index), ncols=nums, sharey=True, sharex=True)
    if len(index) == 1:
        for ax,img in zip(axes.flatten(),IMG[index[0]]):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.imshow(img)
    else:
        for ax_row, idx in zip(axes, index):
            img_row = IMG[idx][0:nums]
            for img,ax in zip(img_row,ax_row):
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.imshow(img)
    fig.tight_layout(pad=0)
    plt.show()

def NORMALIZATION(data):
    from sklearn.preprocessing import MinMaxScaler
    minmax = MinMaxScaler()
    data2 = minmax.fit_transform(data)
    return data2

def DataBatch(data,label,dataSize,labelSize,isShuffle,batchSize):

    capacity = 500 
    data.set_shape(dataSize)
    label.set_shape(labelSize)
    min_after_dequeue = 2 * batchSize 

    if isShuffle:
        [data_batch, label_batch] = tf.train.shuffle_batch([data,label],batch_size=batchSize,capacity=capacity,
                                                       min_after_dequeue=min_after_dequeue)
    else:
        [data_batch,label_batch] = tf.train.batch([data,label],batch_size=batchSize,capacity=capacity)

    return [data_batch,label_batch]

def ReadFromTFRecord(sameName,isShuffle,datatype,labeltype,isMultithreading):

    fileslist = tf.train.match_filenames_once(sameName)
    filename_queue = tf.train.string_input_producer(fileslist,shuffle=isShuffle)
    reader = tf.TFRecordReader()
    _,serialization = reader.read(filename_queue)
    if isMultithreading:
        qr = tf.train.QueueRunner(filename_queue, [serialization] * 7)
        tf.train.add_queue_runner(qr)
    features = tf.parse_single_example(
        serialization,
        features={
            "data": tf.FixedLenFeature([],tf.string), 
            "label": tf.FixedLenFeature([],tf.string) 
        })

    data = tf.decode_raw(features["data"], datatype)
    label = tf.decode_raw(features['label'], labeltype)

    return [data,label]

def Int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def Bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def SaveByTFRecord(data,label,filename,npart):
    
    dir,file = os.path.split(filename)
    if not os.path.isdir(dir):
        print('build directory..'%(dir))
        os.mkdir(dir)

    index = np.int32(np.linspace(0,data.shape[0],npart+1))

    for i in range(npart):

        suffix = "-%.1d-of-%.1d"%(i+1,npart)
        newname = filename + suffix

        writer = tf.python_io.TFRecordWriter(newname)

        for j in range(index[i],index[i+1]):

            data_to_string = data[j].tobytes()
            label_to_string = label[j].tobytes()
            feature = {

                "data": Bytes_feature(data_to_string),
                "label": Bytes_feature(label_to_string),

            }
            features = tf.train.Features(feature=feature) 
            example = tf.train.Example(features=features)

            writer.write(example.SerializeToString())
        writer.close()
    fileSets = os .listdir(dir)
    print(fileSets)

def GetCifar10Data(CifarPath, kind):

    fo = open(CifarPath, 'rb')
    cifar10_dict  = pickle.load(fo, encoding='bytes')
    cifar10_label = cifar10_dict.get(b'labels')
    cifar10_data = cifar10_dict.get(b'data')
    L = [label for label in cifar10_label if label == kind]
    C = [cifar10_data[label[0]] for label in enumerate(cifar10_label) if label[1] == kind]
    C = np.array(C)
    L = np.array(L)
    fo.close()

    return C,L

def GetCifar10AllData(kind):
    
    C, L = GetCifar10Data(r'./cifar-10-batches-py/data_batch_1', kind)
    for i in range(2, 6):
        filename = './/cifar-10-batches-py//data_batch_' + str(i)
        data, label = GetCifar10Data(filename, kind)
        C = np.concatenate((C, data))
        L = np.concatenate((L, label))
    return C,L

if __name__ == '__main__':
    C,L = GetCifar10AllData(1)
    C = NORMALIZATION(C)

    imgs = C[-26:-1].reshape(-1,3,32,32).transpose((0,2,3,1))
    fig, axes = plt.subplots(figsize=(20, 7), nrows=5, ncols=5, sharex=True, sharey=True)
    for ax,img in zip(axes.flatten(),imgs):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(img)
    plt.show()

    SaveByTFRecord(C,L,r'./TFR/class1',5)