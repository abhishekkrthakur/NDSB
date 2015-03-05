import skimage.transform as transform
import skimage

IMG_DIM = 96

AUGMENTATION_PARAMS = {
            'zoom_range': (1.0, 1.1),
            'rotation_range': (0, 360),
            'shear_range': (0, 5),
            'translation_range': (-4, 4),
        }

def fast_warp(img, tf, output_shape=IMG_DIM, mode='nearest'):
    """
    This wrapper function is about five times faster than skimage.transform.warp, for our use case.
    """
    m = tf.params
    img_wf = np.empty((output_shape[0], output_shape[1]), dtype='float32')
    img_wf = skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode)
    return img_wf

def random_perturbation_transform(zoom_range, rotation_range, shear_range, translation_range, do_flip=False):
    # random shift [-4, 4] - shift no longer needs to be integer!
    shift_x = np.random.uniform(*translation_range)
    shift_y = np.random.uniform(*translation_range)
    translation = (shift_x, shift_y)

    # random rotation [0, 360]
    rotation = np.random.uniform(*rotation_range) # there is no post-augmentation, so full rotations here!

    # random shear [0, 5]
    shear = np.random.uniform(low=shear_range[0], high=shear_range[1])

    # random zoom [0.9, 1.1]
    # zoom = np.random.uniform(*zoom_range)
    log_zoom_range = [np.log(z) for z in zoom_range]
    zoom = np.exp(np.random.uniform(*log_zoom_range)) # for a zoom factor this sampling approach makes more sense.
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

    # flip
    if do_flip:
        shear += 180
        rotation += 180

    return build_augmentation_transform(zoom, rotation, shear, translation)


def build_augmentation_transform(zoom=1.0, rotation=0, shear=0, translation=(0, 0)):
    center_shift = np.array(IMG_DIM) / 2. - 0.5
    tform_center = transform.SimilarityTransform(translation=-center_shift)
    tform_uncenter = transform.SimilarityTransform(translation=center_shift)

    tform_augment = transform.AffineTransform(scale=(1/zoom, 1/zoom), 
                                              rotation=np.deg2rad(rotation), 
                                              shear=np.deg2rad(shear), 
                                              translation=translation)
    tform = tform_center + tform_augment + tform_uncenter 
    return tform

def transform_randomly(X, plot = False):
    X = X.copy()
    tform_augment = random_perturbation_transform(**AUGMENTATION_PARAMS)
    tform_identity = skimage.transform.AffineTransform()
    tform_ds = skimage.transform.AffineTransform()

    for i in range(X.shape[0]):
        new = fast_warp(X[i][0], tform_ds + tform_augment + tform_identity, 
                             output_shape=IMG_DIM, mode='nearest').astype('float32')
        X[i,:] = new

    return X


#################### Replaces old BatchIterator ##########################

from nolearn.lasagne import BatchIterator

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        Xb = transform_randomly(Xb)
        return Xb, yb
