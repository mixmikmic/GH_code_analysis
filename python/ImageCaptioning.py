from examples.ImageCaption import *
from pyspark.sql import SQLContext
from pyspark import SparkConf,SparkContext
from itertools import izip_longest
from com.yahoo.ml.caffe.tools.DFConversions import *
from com.yahoo.ml.caffe.tools.Vocab import *
from com.yahoo.ml.caffe.Config import *
from com.yahoo.ml.caffe.CaffeOnSpark import *
from com.yahoo.ml.caffe.Config import *
from com.yahoo.ml.caffe.DataSource import *
import json

conv=DFConversions(sc)
df_image_caption=conv.Coco2ImageCaptionFile("/tmp/coco/annotations/captions_train2014.json",1)
vocab=Vocab(sc)
vocab.genFromData(df_image_caption,"caption",8800)
df_embedding = conv.ImageCaption2Embedding("/tmp/coco/images/train2014", df_image_caption, vocab,20)
df_embedding.write.parquet("/tmp/coco/parquet/df_embedded_train2014")

cos=CaffeOnSpark(sc)
args={}
args['conf']='CaffeOnSpark/data/bvlc_reference_solver.prototxt'
args['model']='file:///tmp/coco/bvlc_reference_caffenet.caffemodel'
args['devices']='1'
args['clusterSize']='1'
cfg=Config(sc,args)
dl_train_image = DataSource(sc).getSource(cfg,True)
cos.train(dl_train_image)

args={}
args['conf']='CaffeOnSpark/data/lrcn_solver.prototxt'
args['model']='file:///tmp/coco/parquet/lrcn_coco.model'
args['devices']='1'
args['clusterSize']='1'
args['weights']='/tmp/coco/bvlc_reference_caffenet.caffemodel'
args['resize']='True'
cfg=Config(sc,args)
dl_train_lstm = DataSource(sc).getSource(cfg,True)
cos.train(dl_train_lstm)

conv=DFConversions(sc)
vocab=Vocab(sc)
df_image_caption_test=conv.Coco2ImageCaptionFile("/tmp/coco/annotations/captions_demo.json",1)
vocab.genFromData(df_image_caption_test,"caption",8800)
df_embedding = conv.ImageCaption2Embedding("/tmp/coco/images/demo/", df_image_caption_test, vocab,20)
df_embedding.write.parquet("/tmp/coco/parquet/df_embedded_test")
df_embedded_input = sqlContext.read.parquet("/tmp/coco/parquet/df_embedded_test")

df_images = df_embedded_input.select("data.image","data.height", "data.width", "id")
model="/tmp/coco/parquet/lrcn_coco.model"
imagenet="CaffeOnSpark/data/lstm_deploy.prototxt"
lstmnet="CaffeOnSpark/data/lrcn_word_to_preds.deploy.prototxt"
vocab="CaffeOnSpark/data/vocab.txt"

df_results = get_predictions(sqlContext,df_images,model,imagenet,lstmnet,vocab)
df_results.show()

df=df_embedded_input.join(df_results, df_embedded_input.id == df_results.id)

from com.yahoo.ml.caffe.DisplayUtils import *
show_captions(df)



