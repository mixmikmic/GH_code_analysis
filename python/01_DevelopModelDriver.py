get_ipython().run_cell_magic('writefile', 'driver.py', 'import base64\nimport json\nimport logging\nimport os\nimport timeit as t\nfrom io import BytesIO\n\nimport numpy as np\nimport tensorflow as tf\nfrom PIL import Image, ImageOps\nfrom tensorflow.contrib.slim.nets import resnet_v1\n\n_MODEL_FILE = os.getenv(\'MODEL_FILE\', "resnet_v1_152.ckpt")\n_LABEL_FILE = os.getenv(\'LABEL_FILE\', "synset.txt")\n_NUMBER_RESULTS = 3\n\n\ndef _create_label_lookup(label_path):\n    with open(label_path, \'r\') as f:\n        label_list = [l.rstrip() for l in f]\n        \n    def _label_lookup(*label_locks):\n        return [label_list[l] for l in label_locks]\n    \n    return _label_lookup\n\n\ndef _load_tf_model(checkpoint_file):\n    # Placeholder\n    input_tensor = tf.placeholder(tf.float32, shape=(None,224,224,3), name=\'input_image\')\n    \n    # Load the model\n    sess = tf.Session()\n    arg_scope = resnet_v1.resnet_arg_scope()\n    with tf.contrib.slim.arg_scope(arg_scope):\n        logits, _ = resnet_v1.resnet_v1_152(input_tensor, num_classes=1000, is_training=False, reuse=tf.AUTO_REUSE)\n    probabilities = tf.nn.softmax(logits)\n    \n    saver = tf.train.Saver()\n    saver.restore(sess, checkpoint_file)\n    \n    def predict_for(image):\n        pred, pred_proba = sess.run([logits,probabilities], feed_dict={input_tensor: image})\n        return pred_proba\n    \n    return predict_for\n\n\ndef _base64img_to_numpy(base64_img_string):\n    if base64_img_string.startswith(\'b\\\'\'):\n        base64_img_string = base64_img_string[2:-1]\n    base64Img = base64_img_string.encode(\'utf-8\')\n\n    # Preprocess the input data \n    startPreprocess = t.default_timer()\n    decoded_img = base64.b64decode(base64Img)\n    img_buffer = BytesIO(decoded_img)\n\n    # Load image with PIL (RGB)\n    pil_img = Image.open(img_buffer).convert(\'RGB\')\n    pil_img = ImageOps.fit(pil_img, (224, 224), Image.ANTIALIAS)\n    return np.array(pil_img, dtype=np.float32)\n\n\ndef create_scoring_func(model_path=_MODEL_FILE, label_path=_LABEL_FILE):\n    logger = logging.getLogger("model_driver")\n    \n    start = t.default_timer()\n    labels_for = _create_label_lookup(label_path)\n    predict_for = _load_tf_model(model_path)\n    end = t.default_timer()\n\n    loadTimeMsg = "Model loading time: {0} ms".format(round((end-start)*1000, 2))\n    logger.info(loadTimeMsg)\n    \n    def call_model(image_array, number_results=_NUMBER_RESULTS):\n        pred_proba = predict_for(image_array).squeeze()\n        selected_results = np.flip(np.argsort(pred_proba), 0)[:number_results]\n        labels = labels_for(*selected_results)\n        return list(zip(labels, pred_proba[selected_results].astype(np.float64)))\n    return call_model\n\n\ndef get_model_api():\n    logger = logging.getLogger("model_driver")\n    scoring_func = create_scoring_func()\n    \n    def process_and_score(images_dict, number_results=_NUMBER_RESULTS):\n        start = t.default_timer()\n\n        results = {}\n        for key, base64_img_string in images_dict.items():\n            rgb_image = _base64img_to_numpy(base64_img_string)\n            batch_image = np.expand_dims(rgb_image, 0)\n            results[key]=scoring_func(batch_image, number_results=_NUMBER_RESULTS)\n        \n        end = t.default_timer()\n\n        logger.info("Predictions: {0}".format(results))\n        logger.info("Predictions took {0} ms".format(round((end-start)*1000, 2)))\n        return (results, \'Computed in {0} ms\'.format(round((end-start)*1000, 2)))\n    return process_and_score\n\ndef version():\n    return tf.__version__\n    ')

import logging

logging.basicConfig(level=logging.DEBUG)

get_ipython().run_line_magic('run', 'driver.py')

from testing_utilities import img_url_to_json

IMAGEURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Lynx_lynx_poing.jpg/220px-Lynx_lynx_poing.jpg"

jsonimg = img_url_to_json(IMAGEURL)

json_lod= json.loads(jsonimg)

predict_for = get_model_api()

output = predict_for(json_lod['input'])

json.dumps(output)

