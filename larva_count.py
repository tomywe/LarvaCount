from PIL import Image
import os
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import wget

from model import CSRNet

def count(path):
    """
    evaluates the number of larva present in input.
    input is either an image of a video. if input is an image, the evaluation is done once over the image, if input is a
    video, the evaluation is done over every caption in the video seperately and then averaged over all captions to
    produce the result
    :param path: a path to an image or a video
    :return: count
    """


    # Define the device(processor) type
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Current procesor is GPU')
    else:
        device = torch.device('cpu')
        print('Current procesor is CPU')

    # Define the model to use for calculations
    model = CSRNet()
    model.load_state_dict(torch.load('model_wgts.pth'))
    model.to(device)
    model.eval()

    # Load the image or video
    im_list = []
    try:
        im_list.append(Image.open(path))
    except OSError:
        if 'http' in path:
            wget.download(path, out='videos')
        cap = cv2.VideoCapture(os.listdir('videos')[0])
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fc = 0
        ret = True
        im_list = []
        while (fc < frameCount and ret):
            ret, im = cap.read()
            if fc%10 == 0:
                new_im = np.zeros_like(im)
                new_im[:,:,0] = im[:,:,2]
                new_im[:,:,1] = im[:,:,1]
                new_im[:,:,2] = im[:,:,0]
                im_list.append(Image.fromarray(new_im.astype('uint8'), 'RGB'))
            fc += 1

    # Disable gradients
    with torch.no_grad():
        # Prepare data for model
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_eval = T.Compose([T.Resize(255, interpolation=Image.BICUBIC), T.ToTensor(), T.Normalize(mean, std)])
        model_input = torch.stack([transform_eval(im) for im in im_list])
        model_input.to(device)

        results, densities = model(model_input)
        if len(results) > 1:
            results = results.mean()
        return results
