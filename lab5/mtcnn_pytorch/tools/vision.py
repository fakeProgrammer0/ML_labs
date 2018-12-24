import matplotlib.pyplot as plt


def vis_face(im_array, dets, landmarks=None):
    """Visualize detection results of an image

    Parameters:
    ----------
    im_array: numpy.ndarray, shape(1, c, h, w)
        test image in rgb
    dets: numpy.ndarray([[x1 y1 x2 y2 score landmarks]])
        detection results before calibration
    landmarks: numpy.ndarray([landmarks for five facial landmarks])

    Returns:
    -------
    """
    figure = plt.figure()
    plt.imshow(im_array)
    figure.suptitle('Face Detector', fontsize=12, color='r')

    for i in range(dets.shape[0]):
        bbox = dets[i, 0:4]
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor='yellow', linewidth=0.9)
        plt.gca().add_patch(rect)

    if landmarks is not None:
        for i in range(landmarks.shape[0]):
            landmarks_one = landmarks[i, :]
            landmarks_one = landmarks_one.reshape((5, 2))
            for j in range(5):
                plt.scatter(landmarks_one[j, 0], landmarks_one[j, 1], c='red', linewidths=1, marker='x', s=5)

    plt.show()
