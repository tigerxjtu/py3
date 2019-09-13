# coding:utf-8
'''
V3.0A版本，尝试实现摄像头识别
'''
import numpy as np
import cv2
import os
import os.path
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pylab
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class UiForm():
    openfile_name_pb = ''
    openfile_name_pbtxt = ''
    openpic_name = ''
    num_class = 0

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(600, 690)
        Form.setMinimumSize(QtCore.QSize(600, 690))
        Form.setMaximumSize(QtCore.QSize(600, 690))
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setGeometry(QtCore.QRect(20, 20, 550, 100))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        # 加载模型文件按钮
        self.btn_add_file = QtWidgets.QPushButton(self.frame)
        self.btn_add_file.setObjectName("btn_add_file")
        self.horizontalLayout_2.addWidget(self.btn_add_file)
        # 加载pbtxt文件按钮
        self.btn_add_pbtxt = QtWidgets.QPushButton(self.frame)
        self.btn_add_pbtxt.setObjectName("btn_add_pbtxt")
        self.horizontalLayout_2.addWidget(self.btn_add_pbtxt)
        # 输入检测类别数目按钮
        self.btn_enter = QtWidgets.QPushButton(self.frame)
        self.btn_enter.setObjectName("btn_enter")
        self.horizontalLayout_2.addWidget(self.btn_enter)
        # 打开摄像头
        self.btn_opencam = QtWidgets.QPushButton(self.frame)
        self.btn_opencam.setObjectName("btn_objdec")
        self.horizontalLayout_2.addWidget(self.btn_opencam)
        # 开始识别按钮
        self.btn_objdec = QtWidgets.QPushButton(self.frame)
        self.btn_objdec.setObjectName("btn_objdec")
        self.horizontalLayout_2.addWidget(self.btn_objdec)
        # 退出按钮
        self.btn_exit = QtWidgets.QPushButton(self.frame)
        self.btn_exit.setObjectName("btn_exit")
        self.horizontalLayout_2.addWidget(self.btn_exit)
        # 显示识别后的画面
        self.lab_rawimg_show = QtWidgets.QLabel(Form)
        self.lab_rawimg_show.setGeometry(QtCore.QRect(50, 140, 500, 500))
        self.lab_rawimg_show.setMinimumSize(QtCore.QSize(500, 500))
        self.lab_rawimg_show.setMaximumSize(QtCore.QSize(500, 500))
        self.lab_rawimg_show.setObjectName("lab_rawimg_show")
        self.lab_rawimg_show.setStyleSheet(("border:2px solid red"))


        self.retranslateUi(Form)
        # 这里将按钮和定义的动作相连，通过click信号连接openfile槽？
        self.btn_add_file.clicked.connect(self.openpb)
        # 用于打开pbtxt文件
        self.btn_add_pbtxt.clicked.connect(self.openpbtxt)
        # 用于用户输入类别数
        self.btn_enter.clicked.connect(self.enter_num_cls)
        # 打开摄像头
        self.btn_opencam.clicked.connect(self.opencam)
        # 开始识别
        # ~ self.btn_objdec.clicked.connect(self.object_detection)
        # 这里是将btn_exit按钮和Form窗口相连，点击按钮发送关闭窗口命令
        self.btn_exit.clicked.connect(Form.close)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "目标检测"))
        self.btn_add_file.setText(_translate("Form", "加载模型文件"))
        self.btn_add_pbtxt.setText(_translate("Form", "加载pbtxt文件"))
        self.btn_enter.setText(_translate("From", "指定识别类别数"))
        self.btn_opencam.setText(_translate("Form", "打开摄像头"))
        self.btn_objdec.setText(_translate("From", "开始识别"))
        self.btn_exit.setText(_translate("Form", "退出"))
        self.lab_rawimg_show.setText(_translate("Form", "识别效果"))

    def openpb(self):
        global openfile_name_pb
        openfile_name_pb, _ = QFileDialog.getOpenFileName(self.btn_add_file,'选择pb文件','/home/kanghao/','pb_files(*.pb)')
        print('加载模型文件地址为：' + str(openfile_name_pb))

    def openpbtxt(self):
        global openfile_name_pbtxt
        openfile_name_pbtxt, _ = QFileDialog.getOpenFileName(self.btn_add_pbtxt,'选择pbtxt文件','/home/kanghao/','pbtxt_files(*.pbtxt)')
        print('加载标签文件地址为：' + str(openfile_name_pbtxt))

    def opencam(self):
        self.camcapture = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer()
        self.timer.start()
        self.timer.setInterval(100) # 0.1s刷新一次
        self.timer.timeout.connect(self.camshow)

    def camshow(self):
        global camimg
        _ , camimg = self.camcapture.read()
        print(_)
        camimg = cv2.resize(camimg, (512, 512))
        camimg = cv2.cvtColor(camimg, cv2.COLOR_BGR2RGB)
        print(type(camimg))
        #strcamimg = camimg.tostring()
        showImage = QtGui.QImage(camimg.data, camimg.shape[1], camimg.shape[0], QtGui.QImage.Format_RGB888)
        self.lab_rawimg_show.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def enter_num_cls(self):
        global num_class
        num_class, okPressed = QInputDialog.getInt(self.btn_enter,'指定训练类别数','你的目标有多少类？',1,1,28,1)
        if okPressed:
            print('识别目标总类为：' + str(num_class))

    def img2pixmap(self, image):
        Y, X = image.shape[:2]
        self._bgra = np.zeros((Y, X, 4), dtype=np.uint8, order='C')
        self._bgra[..., 0] = image[..., 2]
        self._bgra[..., 1] = image[..., 1]
        self._bgra[..., 2] = image[..., 0]
        qimage = QtGui.QImage(self._bgra.data, X, Y, QtGui.QImage.Format_RGB32)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        return pixmap

    def object_detection(self):
        sys.path.append("..")
        from object_detection.utils import ops as utils_ops

        if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
            raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

        from utils import label_map_util

        from utils import visualization_utils as vis_util

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_FROZEN_GRAPH = openfile_name_pb

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = openfile_name_pbtxt

        NUM_CLASSES = num_class

        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

        def load_image_into_numpy_array(image):
          (im_width, im_height) = image.size
          return np.array(image.getdata()).reshape(
              (im_height, im_width, 3)).astype(np.uint8)

        # For the sake of simplicity we will use only 2 images:
        # image1.jpg
        # image2.jpg
        # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
        TEST_IMAGE_PATHS = camimg
        print(TEST_IMAGE_PATHS)
        # Size, in inches, of the output images.
        IMAGE_SIZE = (12, 8)

        def run_inference_for_single_image(image, graph):
          with graph.as_default():
            with tf.Session() as sess:
              # Get handles to input and output tensors
              ops = tf.get_default_graph().get_operations()
              all_tensor_names = {output.name for op in ops for output in op.outputs}
              tensor_dict = {}
              for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
              ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                  tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)
              if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
              image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

              # Run inference
              output_dict = sess.run(tensor_dict,
                                     feed_dict={image_tensor: np.expand_dims(image, 0)})

              # all outputs are float32 numpy arrays, so convert types as appropriate
              output_dict['num_detections'] = int(output_dict['num_detections'][0])
              output_dict['detection_classes'] = output_dict[
                  'detection_classes'][0].astype(np.uint8)
              output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
              output_dict['detection_scores'] = output_dict['detection_scores'][0]
              if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
          return output_dict


        #image = Image.open(TEST_IMAGE_PATHS)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(TEST_IMAGE_PATHS)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        #plt.savefig(str(TEST_IMAGE_PATHS)+".jpg")

## 用于显示ui界面的命令
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Window = QtWidgets.QWidget()
    # ui为根据类Ui_From()创建的实例
    ui = UiForm()
    ui.setupUi(Window)
    Window.show()
    sys.exit(app.exec_())