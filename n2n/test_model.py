import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_unet_model
from noise_model_test import get_noise_model


def get_args(input_args):
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--network_depth", type=int, default=4,
                        help="encoder-decoder network depth")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, required=True,
                        help="trained weight file")
    parser.add_argument("--test_noise_model", type=str, default="text,0,25",
                        help="noise model for test images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="if set, save resulting images otherwise show result using imshow")
    args = parser.parse_args(input_args)
    return args


def get_image(image):
    return np.clip(image, 0, 3071-(-1024))


# CT need to be transformed from uint16 to int16
# HU values are from [-1204 : 3071]
def ct_intensity_to_HU(image):
    im = image.astype(np.float32, copy=False) - 32768
    return (im - (-1024)).astype(np.int16, copy=False)


# CT need to be transformed from int16 to uint16
# HU values are from [-1204 : 3071]
def HU_to_ct_intensity(image):
    im = image.astype(np.float32, copy=False) - 1024
    return (im + 32768).astype(np.uint16, copy=False)


def main(*input_args):
    args = get_args(input_args)
    image_dir = args.image_dir
    weight_file = args.weight_file
    net_depth = args.network_depth
    model = get_unet_model(depth=net_depth)
    model.load_weights(weight_file)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(image_dir).glob("*.*"))

    for image_path in image_paths:
        image = cv2.imread(str(image_path), -1)
        h, w, _ = image.shape

        image_HU = ct_intensity_to_HU(image)
        pred = model.predict(np.expand_dims(image_HU, 0))
        denoised_image = HU_to_ct_intensity(get_image(pred[0]))

        if args.output_dir:
            cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".png", denoised_image)
        else:
            cv2.imshow("result", denoised_image)
            key = cv2.waitKey(-1)
            # "q": quit
            if key == 113:
                return 0


if __name__ == '__main__':
    main()
