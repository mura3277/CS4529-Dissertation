# Prerequisites
from numpy import ndarray
from cv2 import imshow, waitKey, destroyAllWindows, imread, imwrite

def show_image(source,**kwargs):
    if isinstance(source,str):
        im=imread(source)
    elif isinstance(source, ndarray):
        im=source
    else:
        raise TypeError("Source Not of Correct Type")
        return None
    imshow(None,im)
    waitKey(0)
    destroyAllWindows()
    if "file_path" in kwargs:
        imwrite(kwargs["file_path"],im)
