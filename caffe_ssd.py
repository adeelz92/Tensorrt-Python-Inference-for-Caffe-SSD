import cv2 as cv
import pycuda.driver as cuda
import tensorrt as trt
import common
import numpy as np

class ModelData(object):
    MODEL_PATH = "./caffe_model/VGG_VOC0712_SSD_300x300_iter_8000.caffemodel"
    DEPLOY_PATH = "./caffe_model/deploy_jetson.prototxt"
    INPUT_SHAPE = (3, 300, 300)
    OUTPUT_NAME1 = "detection_out"
    OUTPUT_NAME2 = "keep_count"
    DTYPE = trt.float32

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list
def load_normalized_test_case(test_image, pagelocked_buffer):

    def normalize_image(cv_image, mean_data=[104,117,123]):
        c, h, w = ModelData.INPUT_SHAPE
        image_arr = cv.resize(cv_image, (h, w))
        image_arr = np.float32(image_arr)
        mean = np.array(mean_data, dtype=np.float32)
        image_arr -= mean
        image_arr = image_arr.transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()
        print(image_arr.shape)
        return image_arr

    np.copyto(pagelocked_buffer, normalize_image(test_image))
    return test_image


def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def do_inference(context, h_input, d_input, h_output, d_output, stream):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

def build_engine_caffe(model_file, deploy_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
        builder.max_workspace_size = common.GiB(1)
        model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)
        # print(model_tensors)
        network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME1))
        # network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME2))
        return builder.build_cuda_engine(network)

def ScaleBoundingbox(bbox,width, height):
    bbox[0] *= width
    bbox[1] *= height
    bbox[2] *= width
    bbox[3] *= height
    return bbox

def vis_detections(im, class_name, bbox, score):
    cv.rectangle(im,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(255,0,0),2)
    cv.putText(im, "{:s} {:.3f}".format(class_name, score), (bbox[0], (int)(bbox[1]+10)),
        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

all_classes = ('__background__', 'red_and_white_bomb','blue_object', 'blue_box','brick','red_balls_packet',
               'black_box','black_packet', 'red_packet', 'bottle','knife','cigerate','silver_plate','chop_sticks',
               'knife2','red_cloth', 'silver_sheet')

def main(test_image):
    caffe_model_file = ModelData.MODEL_PATH
    caffe_deploy_file = ModelData.DEPLOY_PATH
    height = test_image.shape[0]
    width = test_image.shape[1]
    engine = build_engine_caffe(caffe_model_file, caffe_deploy_file)
    h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
    context = engine.create_execution_context()
    test_case = load_normalized_test_case(test_image.copy(), h_input)
    do_inference(context, h_input, d_input, h_output, d_output, stream)
    h_output = h_output.reshape(-1, 7)
    h_output = h_output[:, 1:]
    # for h in h_output:
    #     print(h)
    cls_index = h_output[:, 0]
    cls_scores = h_output[:, 1]
    cls_boxes = h_output[:, 2:]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    threshhold = 0.10
    inds = np.where(dets[:, -1] >= threshhold)[0]
    for i in inds:
        bbox = dets[i, :4]
        bbox = ScaleBoundingbox(bbox, width, height)
        # print("original_boxes", bbox)
        object_width = int(bbox[2] - bbox[0])
        object_height = int(bbox[3] - bbox[1])
        x = int(bbox[0])
        y = int(bbox[1])
        box = [x, y, object_width, object_height]

        # print("Boxes", box)
        score = dets[i, -1] * 100
        vis_detections(test_image, all_classes[int(cls_index[i])], bbox, score)
        print("Score", score)
        print("Box", box)
        cv.imwrite("NewImage.jpg", test_image)

if __name__ == '__main__':
    img = cv.imread("17-6.jpg")
    main(img)