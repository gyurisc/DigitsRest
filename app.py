import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil

import caffe

from flask import jsonify

from classify_cli import classify
from classify_cli import get_net
from classify_cli import get_transformer
from classify_cli import forward_pass
from classify_cli import read_labels

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
APP_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
MODEL_FOLDER = os.path.abspath(APP_DIRNAME + '/model')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])


print('Configured Folders:')
print(APP_DIRNAME)
print(MODEL_FOLDER)
print(UPLOAD_FOLDER)

# Obtain the flask app object
app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)
    
@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result, imagesrc=imageurl)
        
def save_image(imagefile):
    filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
        werkzeug.secure_filename(imagefile.filename)
    filename = os.path.join(UPLOAD_FOLDER, filename_)
    imagefile.save(filename)
    logging.info('Saving to %s.', filename)
    image = exifutil.open_oriented_im(filename)
    return (filename, image)
         
@app.route('/api/classify_upload', methods=['POST'])
def rest_classify_upload():     
    try:
        imagefile = flask.request.files['imagefile']
        filename, image = save_image(imagefile)        
    except Exception as err: 
        logging.info('Uploaded image open error: %s', err)
        return jsonify({'result' : (False, 'Cannot open uploaded image.')})
        
    print('classifying image %s' % filename)
    result = app.clf.classify_image(filename)    
    return jsonify({'result' : result})    
        
@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename, image = save_image(imagefile)
        
        #filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
        #    werkzeug.secure_filename(imagefile.filename)
        #filename = os.path.join(UPLOAD_FOLDER, filename_)
        #imagefile.save(filename)
        #logging.info('Saving to %s.', filename)
        #image = exifutil.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    print('classifying image %s' % filename)
    result = app.clf.classify_image(filename)
    ## result = None
    return flask.render_template(
        'index.html', has_result=True, result=result,
        imagesrc=embed_image_html(image)
    )


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )

class ClassifierModel(object):
    default_args = {
        'caffe_model' : None, 
        'deploy_file' : None, 
        'mean_file' : None,
        'labels_file' : None, 
    }     
    
    # Setting up Model 
    for filename in os.listdir(MODEL_FOLDER):
        full_path = os.path.join(MODEL_FOLDER, filename)
        if filename.endswith('.caffemodel'):
            default_args['caffe_model'] = full_path
        elif filename == 'deploy.prototxt':
            default_args['deploy_file'] = full_path
        elif filename.endswith('.binaryproto'):
            default_args['mean_file'] = full_path
        elif filename == 'labels.txt':
            default_args['labels_file'] = full_path

    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))    
    # assert caffemodel is not None, 'Caffe model file not found'
    # assert deploy_file is not None, 'Deploy file not found'
    
    print('Model initialized OK!')             
    
    def __init__(self, caffe_model, deploy_file, mean_file, labels_file, gpu_mode):
        logging.info('Loading model and associated files...')
        self.caffe_model = caffe_model
        self.deploy_file = deploy_file
        self.mean_file = mean_file
        self.labels_file = labels_file 
        self.gpu_mode = gpu_mode 
                
    def classify_image(self, image): 
        try:
            print('classify_image %s' %image)     
            starttime = time.time()
            scores = classify(self.caffe_model, self.deploy_file, [image], mean_file = self.mean_file, labels_file = self.labels_file)
            endtime = time.time()
            
            print('Classification took %s second' % (endtime - starttime))                    
            return (True, scores[0], '%.3f' % (endtime - starttime))         
            
        except Exception as err: 
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')
                                           
def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()

def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    ClassifierModel.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf = ClassifierModel(**ClassifierModel.default_args)
    # app.clf.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
