import cv2
import numpy as np
from difflib import SequenceMatcher
from skimage import measure
from sklearn.metrics.pairwise import cosine_similarity
from element_detection.Element import Element


def dhash(image):
    image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    dhash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    return dhash_str


def calc_similarity_hash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return 1 - n / len(hash1)


def calc_similarity_sift_or_surf_or_orb(img1, img2, method, ratio=1.5, draw_match=False):
    if method == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
    elif method == 'surf':
        # Initiate SURF detector
        surf = cv2.xfeatures2d.SURF_create()
        # find the keypoints and descriptors with SURF
        kp1, des1 = surf.detectAndCompute(img1, None)
        kp2, des2 = surf.detectAndCompute(img2, None)
    elif method == 'orb':
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
    else:
        print('set method as either sift or surf')
        return

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0
    matches = bf.knnMatch(des1, des2, k=2)
    # If there's a big difference between the best and second-best matches, this to be a quality match.
    valid_matches = []
    for best, second in matches:
        if second.distance > ratio * best.distance:
            valid_matches.append(best)

    if draw_match:
        board = cv2.drawMatches(img1, kp1, img2, kp2, valid_matches, None)
        cv2.imshow('sift match', cv2.resize(board, (int(board.shape[1] * (400 / board.shape[0])), 400)))
        cv2.waitKey()
        cv2.destroyWindow('sift match')
    return len(valid_matches) / max(len(kp2), len(kp1))


def image_similarity(img1, img2, method='dhash', is_gray=False,
                     draw_match=False, match_distance_ratio=1.5, resnet_model=None):
    '''
    @method: the way to calculate the similarity between two images
        opt - dhash, ssim, sift, surf, resnet
    '''
    similarity = -1  # from 0 to 1
    if method == 'dhash':
        h1 = dhash(img1)
        h2 = dhash(img2)
        similarity = calc_similarity_hash(h1, h2)
    elif method == 'ssim':
        multi_channel = True
        if is_gray:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            multi_channel = False
        if img1.shape != img2.shape:
            side = max(img1.shape[:2] + img2.shape[:2])
            img1_cp = cv2.resize(img1, (side, side))
            img2_cp = cv2.resize(img2, (side, side))
            similarity = measure.compare_ssim(img1_cp, img2_cp, multichannel=multi_channel)
        else:
            similarity = measure.compare_ssim(img1, img2, multichannel=multi_channel)
    elif method == 'sift':
        similarity = calc_similarity_sift_or_surf_or_orb(img1, img2, 'sift', match_distance_ratio, draw_match=draw_match)
    elif method == 'surf':
        similarity = calc_similarity_sift_or_surf_or_orb(img1, img2, 'surf', match_distance_ratio, draw_match=draw_match)
    elif method == 'resnet':
        shape = (32, 32)
        img1 = cv2.resize(img1, shape)
        img2 = cv2.resize(img2, shape)
        if resnet_model is None:
            from keras.applications.resnet50 import ResNet50
            resnet_model = ResNet50(include_top=False, input_shape=(32, 32, 3))
        encodings = resnet_model.predict(np.array([img1, img2]))
        encodings = encodings.reshape((encodings.shape[0], -1))
        similarity = cosine_similarity([encodings[0]], [encodings[1]])[0][0]
    return similarity


def image_similarity_matrix(images1, images2, method='resnet', resnet_model=None):
    '''
    :param images1: a list of image
    :param images2: a list of image
    :param resnet_model: resnet model for encoding
    :param method: the way to calculate the similarity between two images
        opt - dhash, ssim, sift, surf, resnet
    :return: a similarity matrix in shape of (len(images1), len(images2))
    '''
    sim_matrix = []
    if method == 'resnet':
        images_resize = [cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA) for img in images1] + [cv2.resize(img, (32, 32)) for img in images2]
        encodings = resnet_model.predict(np.array(images_resize))
        encodings = encodings.reshape((encodings.shape[0], -1))
        sim_matrix = cosine_similarity(encodings[:len(images1)], encodings[len(images1):])
    elif method == 'dhash':
        for img1 in images1:
            h1 = dhash(img1)
            sim_row = []
            for img2 in images2:
                h2 = dhash(img2)
                sim_row.append(calc_similarity_hash(h1, h2))
            sim_matrix.append(sim_row)
        sim_matrix = np.array(sim_matrix)
    elif method == 'sift':
        for img1 in images1:
            sim_row = []
            for img2 in images2:
                sim_row.append(calc_similarity_sift_or_surf_or_orb(img1, img2, method='sift'))
            sim_matrix.append(sim_row)
        sim_matrix = np.array(sim_matrix)
    elif method == 'surf':
        for img1 in images1:
            sim_row = []
            for img2 in images2:
                sim_row.append(calc_similarity_sift_or_surf_or_orb(img1, img2, method='surf'))
            sim_matrix.append(sim_row)
        sim_matrix = np.array(sim_matrix)
    elif method == 'orb':
        for img1 in images1:
            sim_row = []
            for img2 in images2:
                sim_row.append(calc_similarity_sift_or_surf_or_orb(img1, img2, method='orb'))
            sim_matrix.append(sim_row)
        sim_matrix = np.array(sim_matrix)
    elif method == 'template-match':
        sim_matrix = np.array(sim_matrix)
    return sim_matrix


def text_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()


def match_element_template_matching(gui_img, target_ele_img, show=False):
    img_gray = cv2.cvtColor(gui_img, cv2.COLOR_BGR2GRAY)
    target_img_gray = cv2.cvtColor(target_ele_img, cv2.COLOR_BGR2GRAY)
    w, h = target_img_gray.shape[::-1]

    # methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
    method = cv2.TM_CCOEFF_NORMED
    res = cv2.matchTemplate(img_gray, target_img_gray, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    element = Element(element_id=100, category='Compo', position={'column_min':top_left[0], 'row_min':top_left[1], 'column_max':bottom_right[0], 'row_max':bottom_right[1]})
    if show:
        board = gui_img.copy()
        element.draw_element(board, show=True)
    return element
