import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
import itertools
import argparse
from PIL import Image
import sys


def get_args():
    # 使用可能なオプションの指定
    parser = argparse.ArgumentParser(description='顔交換プログラム')
    parser.add_argument('--target', '-t', default='./sample2.png', help='使用する画像ファイルの指定')
    parser.add_argument('--save', '-s', action='store_true', default=False, help='出力結果を保存するするかどうか')
    parser.add_argument('--outname', '-o', default='./result.png', help='出力結果を保存する際のファイル名の指定')
    parser.add_argument('--debug', default=False, action='store_true', help='結果を出力するかどうか')

    args = parser.parse_args()
    return args

args = get_args()

def output_to_window(img):
    # opencvのBGR形式からPillowでの出力用にRGBへ変換
    img_result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # numpy.ndarray形式からPillowのImageオブジェクトへ変換
    pil_img = Image.fromarray(img_result.astype(np.uint8))
    # プレビューへ出力(macの場合)
    pil_img.show()

# 引用:https://qiita.com/pokohide/items/43203f109fd95df9a7cc
def transformation_from_points(t_points, o_points):
    t_points = t_points.astype(np.float64)
    o_points = o_points.astype(np.float64)

    t_mean = np.mean(t_points, axis = 0)
    o_mean = np.mean(o_points, axis = 0)

    t_points -= t_mean
    o_points -= o_mean

    t_std = np.std(t_points)
    o_std = np.std(o_points)

    t_points -= t_std
    o_points -= o_std

    # https://qiita.com/kyoro1/items/4df11e933e737703d549
    U, S, Vt = np.linalg.svd(t_points.T * o_points)
    R = (U * Vt).T

    return np.vstack(
        [np.hstack((( o_std / t_std ) * R, o_mean.T - ( o_std / t_std ) * R * t_mean.T )),
        np.matrix([ 0., 0., 1. ])]
    )

# 引用:https://qiita.com/pokohide/items/43203f109fd95df9a7cc
def warp_image(image, M, dshape):
    output_image = np.zeros(dshape, dtype = image.dtype)
    cv2.warpAffine(
        image,
        M[:2],
        (dshape[1], dshape[0]),
        dst = output_image, borderMode = cv2.BORDER_TRANSPARENT, flags = cv2.WARP_INVERSE_MAP
    )
    return output_image

def get_feature(img):
    detector = dlib.get_frontal_face_detector()
    rects = detector(img, 1)
    PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
    PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)

    landmarks = []
    print("getting face feature...")
    for rect in rects:
        landmarks.append(np.matrix(
            [[p.x, p.y] for p in PREDICTOR(img, rect).parts()]
        ))
    if len(landmarks) == 2:
        print("get face feature is success")
    elif len(landmarks) == 1:
        print("few faces")
        sys.exit(1)
    elif len(landmarks) > 2:
        print("too many faces")
        sys.exit(1)
    else:
        print("getting face feature is failed")
        sys.exit(1)

    return landmarks

def face_swap(img, landmarks):
    coordinates = []
    for landmark in landmarks:
        coordinates.append(np.array([v for k,v in enumerate(landmark) if k in list(itertools.chain(range(17,27),range(54,61)))]))
    mask = np.zeros_like(img)
    faces = []
    for a_coordinate in coordinates:
        mask = np.zeros_like(img)
        faces.append(mask)
        cv2.fillConvexPoly(faces[-1], points=a_coordinate, color=(255, 255, 255))
    bg_color = (0, 0, 0)
    img2 = np.full_like(img, bg_color)
    results = []
    for a_face in faces:
        results.append(np.where(a_face==255, img, img2))

    M1 = transformation_from_points(landmarks[0][17:],landmarks[1][17:])
    M2 = transformation_from_points(landmarks[1][17:],landmarks[0][17:])
    fit_image1 = warp_image(results[1], M1, img.shape)
    fit_image2 = warp_image(results[0], M2, img.shape)
    result = np.where(fit_image1 == 0, img, fit_image1)
    result = np.where(fit_image2 == 0, result, fit_image2)
    
    return result

def main():
    img = cv2.imread(args.target)
    landmarks = get_feature(img)
    result = face_swap(img, landmarks)

    if args.debug:
        # debugオプションが有効の場合は何も出力しない
        pass
    elif args.save:
        # saveオプションが有効の場合は画像として出力
        cv2.imwrite(args.outname, result)
    else:
        # プレビューへ出力(macの場合)
        output_to_window(result)

if __name__ == '__main__':
    main()
