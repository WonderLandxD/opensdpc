import argparse
import numpy as np
from PIL import Image
import os
import glob


def normalizeStaining(img, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images

    Example use:
        see test.py

    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity

    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''

    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(np.float) + 1) / Io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    # if saveFile is not None:
    #     Image.fromarray(Inorm).save(saveFile)
    #     Image.fromarray(H).save(saveFile)
    #     Image.fromarray(E).save(saveFile)
    #
    #     Image.fromarray(Inorm).save(saveFile + '.png')
    #     Image.fromarray(H).save(saveFile + '_H.png')
    #     Image.fromarray(E).save(saveFile + '_E.png')

    return Inorm, H, E


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--imageFile', type=str, default='example1.tif', help='RGB image file')
    parser.add_argument('--imgList', type=str,
                        default='A:/Datasets/Histopathology/Real_SYSFL_Datasets/UNNORM/val/*/*/*/ROI*.png',
                        help='HE File')
    parser.add_argument('--SaveFile', type=str, default='A:/Datasets/Histopathology/Real_SYSFL_Datasets/NORM_ROI/train',
                        help='save file')
    parser.add_argument('--Io', type=int, default=240)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.15)
    args = parser.parse_args()

    # test_list = glob.glob('/pub/data/chizm/BRACS/BRACS_RoI/resize_version/test/*/*.png')
    # test_save_dir = '/pub/data/chizm/BRACS/BRACS_RoI/norm_version/test'

    img_list = glob.glob(args.imgList)

    for img_path in img_list:
        img_name = img_path.split('\\')[-3] + '_' + img_path.split('\\')[-1]
        class_name = os.path.abspath(os.path.dirname(img_path)).split('\\')[-3]

        save_path = os.path.join(args.SaveFile, class_name, img_name)

        img = np.array(Image.open(img_path))
        normalizeStaining(img=img,
                          Io=args.Io,
                          alpha=args.alpha,
                          beta=args.beta)

        print('{} is done'.format(img_name))

    # img = np.array(Image.open(args.imageFile))
    #
    # normalizeStaining(img = img,
    #                   saveFile = args.saveFile,
    #                   Io = args.Io,
    #                   alpha = args.alpha,
    #                   beta = args.beta)
