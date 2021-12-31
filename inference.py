import tensorflow as tf
import numpy as np
import cv2

# VISUALIZATIONS

def xcycwh_to_xy_min_xy_max(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    # convert the bbox from [xc, yc, w, h] to [xmin, ymin, xmax, ymax].
    bbox_xyxy = tf.concat([bbox[:, :2] - (bbox[:, 2:] / 2), bbox[:, :2] + (bbox[:, 2:] / 2)], axis=-1)
    # Be sure to keep the values btw 0 and 1
    bbox_xyxy = tf.clip_by_value(bbox_xyxy, 0.0, 1.0)
    return bbox_xyxy

def rescale_bbox_xyminxymax(bbox_list, img_shape):
    h, w = img_shape
    
    return bbox_list * [w, h, w, h]

def numpy_bbox_to_image(image, bbox_list, labels=None, scores=None, class_name=[]):
    """ Numpy function used to display the bbox (target or prediction)
    """
    assert(image.dtype == np.float32 and image.dtype == np.float32 and len(image.shape) == 3)
    
    #bbox_x1y1x2y2 = np.concatenate([bbox_list[:, :2], (bbox_list[:, :2] + bbox_list[:, 2:])], axis=-1)
    
    bbox_x1y1x2y2 = rescale_bbox_xyminxymax(xcycwh_to_xy_min_xy_max(bbox_list), (image.shape[0], image.shape[1]))

    # Set the labels if not defined
    if labels is None: labels = np.zeros((bbox_x1y1x2y2.shape[0]))

    bbox_area = []
    # Go through each bbox
    for b in range(0, bbox_x1y1x2y2.shape[0]):
        x1, y1, x2, y2 = bbox_x1y1x2y2[b]
        bbox_area.append((x2-x1)*(y2-y1))

    # Go through each bbox
    for b in np.argsort(bbox_area)[::-1]:
        # Take a new color at random for this instance
        instance_color = np.random.randint(0, 255, (3))
        

        x1, y1, x2, y2 = bbox_x1y1x2y2[b]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)

        # Select the class associated with this bbox
        class_id = labels[int(b)]

        if scores is not None and len(scores) > 0:
            label_name = class_name[int(class_id)]   
            label_name = "%s:%.2f" % (label_name, scores[b])
        else:
            label_name = class_name[int(class_id)]    

        class_color = instance_color[int(class_id)]
    
        color = instance_color
        
        cv2.rectangle(image, (x1, y1), (x2, y2), class_color.tolist(), 2)

    return image




img_for_p=numpy_bbox_to_image(img, tbbox, class_name=['COT'])
img_for_plot = cv2.cvtColor(img_for_p, cv2.COLOR_BGR2RGB)
print(img_for_plot.shape)
plt.figure(figsize=(28, 28))
plt.subplot(121)
plt.imshow(img_for_plot)
plt.title('Image')



def post_process(m_outputs: dict, background_class, bbox_format="xy_center"):

    predicted_bbox = m_outputs["pred_boxes"][0]
    predicted_labels = m_outputs["pred_logits"][0]

    softmax = tf.nn.softmax(predicted_labels)
    predicted_scores = tf.reduce_max(softmax, axis=-1)
    predicted_labels = tf.argmax(softmax, axis=-1)


    indices = tf.where(predicted_labels != background_class)
    indices = tf.squeeze(indices, axis=-1)

    predicted_scores = tf.gather(predicted_scores, indices)
    predicted_labels = tf.gather(predicted_labels, indices)
    predicted_bbox = tf.gather(predicted_bbox, indices)


    if bbox_format == "xy_center":
        predicted_bbox = predicted_bbox
    elif bbox_format == "xyxy":
        predicted_bbox = xcycwh_to_xy_min_xy_max(predicted_bbox)
    else:
        raise NotImplementedError()

    return predicted_bbox, predicted_labels, predicted_scores

def process(m_outputs: dict, background_class, bbox_format="xy_center"):

        predicted_bbox = m_outputs["pred_boxes"]
        predicted_labels = m_outputs["pred_logits"]
        


        softmax = tf.nn.softmax(predicted_labels)
        predicted_scores = tf.reduce_max(softmax, axis=-1)
        
        predicted_labels = tf.math.argmax(softmax, axis=-1)


        indices = tf.where(predicted_labels != background_class)

        predicted_scores = tf.gather_nd(predicted_scores, indices)
        predicted_labels = tf.gather_nd(predicted_labels, indices)
        predicted_bbox = tf.gather_nd(predicted_bbox, indices)


        if bbox_format == "xy_center":
            predicted_bbox = predicted_bbox
        elif bbox_format == "xyxy":
            predicted_bbox = xcycwh_to_xy_min_xy_max(predicted_bbox)
        else:
            raise NotImplementedError()
            
        
        
        return predicted_bbox, predicted_labels, predicted_scores


for valid_images, target_bbox, target_class in valid_iterator:

    m_outputs = detr(valid_images, training=False)
    predicted_bbox, predicted_labels, predicted_scores = process(m_outputs, config.background_class, bbox_format="xy_center")

    result = numpy_bbox_to_image(
        np.array(valid_images[0]),
        np.array(predicted_bbox),
        np.array(predicted_labels),
        scores=np.array(predicted_scores),
        class_name=class_names, 
        config=config
    )
    plt.imshow(result)
    plt.show()
    break


detr_inference = DETR(num_classes=2, num_queries=20)
detr_inference.build()
detr_inference.load_weights('./detr_model.h5')


i = 0
for valid_images, target_bbox, target_class in val_dataset:

    m_outputs = detr_model((valid_images,tf.zeros((tf.shape(valid_images)[0],
                                                   tf.shape(valid_images)[1], tf.shape(valid_images)[2]), tf.bool)), training=False)
    predicted_bbox, predicted_labels, predicted_scores = process(m_outputs, 0, bbox_format="xy_center")

    result = numpy_bbox_to_image(
        np.array(valid_images[0]),
        np.array(predicted_bbox),
        np.array(predicted_labels),
        scores=np.array(predicted_scores),
        class_name=['background', 'COT']
    )
    
    img_for_p=numpy_bbox_to_image(np.array(valid_images[0]), target_bbox[0],class_name=['COT'])
    img_for_plot = cv2.cvtColor(img_for_p, cv2.COLOR_BGR2RGB)
    print(img_for_plot.shape)
    
    plt.imshow(result)
    plt.show()
    
    plt.imshow(img_for_plot)
    plt.title('Image')
    
    break
    
def predict(image_np, detr_model):
    predictions=[]
    m_outputs = detr_model((image_np,tf.zeros((tf.shape(image_np)[0],
                                                   tf.shape(image_np)[1], tf.shape(image_np)[2]), tf.bool)), training=False)
    predicted_bbox, predicted_labels, predicted_scores = process(m_outputs, 0) 
    

    
    #convert to xymin_wh
    predicted_bbox = xcycwh_to_xymin_wh(predicted_bbox)
    
    #rescale bbox based on image size
    predicted_bbox = rescale_bbox_xyminxymax(predicted_bbox, image_np)
    
    
    for i in range(len(predicted_bbox)):
        
        x_min, y_min, bbox_width, bbox_height = predicted_bbox[i]
        
        if predicted_scores[i] >= 0.5:

            predictions.append('{:.2f} {} {} {} {}'.format(predicted_scores[i], x_min, y_min, bbox_width, bbox_height))
    return ''.join(predictions)

i = 0
for valid_images, target_bbox, target_class in val_dataset:
    predictions = predict(tf.expand_dims(valid_images[0], axis=0), detr_inference)
    print(predictions)
    i += 1
    if i > 0:
        break

submission_dict = {
    'row_num': [],
    'annotations': []
}

for (image_np, sample_prediction_df) in iter_test:
    
    sample_prediction_df['annotations'] = predict(tf.expand_dims(image_np, axis=0), detr_inference)

    env.predict(sample_prediction_df)

    submission_dict['row_num'].append(sample_prediction_df['index'].values[0])
    submission_dict['annotations'].append(sample_prediction_df['annotations'].values[0])


pd.DataFrame(submission_dict).to_csv('submission.csv',index=False)
df = pd.read_csv('./submission.csv')

