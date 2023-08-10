import cv2

T1 = cv2.imread('data/templeRing/templeR0001.png', cv2.IMREAD_GRAYSCALE)
sift = cv2.xfeatures2d.SIFT_create(nfeatures=5000)
_, D_1 = sift.detectAndCompute(T1, mask=None)

from ipyparallel import Client
rc = Client()
lview = rc.load_balanced_view()

@lview.parallel()
def get_num_matches(arg):    
    fname, D_src = arg
    import cv2
    frame = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    print frame.shape
    sift = cv2.SIFT(nfeatures=5000)
    _, D = sift.detectAndCompute(frame, mask=None)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(D_src, D)
    return fname, len(matches)

fnames = get_ipython().getoutput('ls data/templeRing/temple*.png')

args = [(fname, D_1) for fname in fnames]
async_res = get_num_matches.map(args)

for f, n in async_res:
    print f, n



