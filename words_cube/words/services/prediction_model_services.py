
import os
# from PIL import Image
import numpy as np
import pickle
import numpy as np
# from PIL import Image as im
from skimage.transform import resize
from skimage.io import imread,imsave
from skimage import data
from skimage.color import rgb2gray
# import matplotlib.pyplot as plt



settings_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.dirname(settings_dir))
path_data = os.path.join(project_root, 'resources\\')
name_model = "allgray.pickle"

Categories=['almostCorrect','correct1','correct2','wrong']
def predict(predict_model_request):
    data = predict_model_request
    data=data.data
    # print("dataaa",data)
    data=data['data']
    print(type(data))
    
    model=pickle.load(open(path_data+name_model,'rb'))
    imsave("out.jpg",np.array(data) )
    # plt.imsave("out.png",np.array(data) )
    img=imread('out.jpg')
   
    img_resize=resize(img,(150,150,3))

    grayscale = rgb2gray(img_resize)

    l=[grayscale.flatten()]
    probability=model.predict_proba(l)
    return {"predict": Categories[model.predict(l)[0]]}



