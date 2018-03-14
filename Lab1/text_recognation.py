from scipy.misc import imsave
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def binarize_image(img_path, target_path, threshold):
    """Binarize an image."""
    image_file = Image.open(img_path)
    image = image_file.convert('L')  # convert image to monochrome
    image = np.array(image)
    image = binarize_array(image, threshold)
    imsave(target_path, image)


def binarize_array(numpy_array, threshold=200):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

#bin_Lenna = 'im_b.png'
#orig_Lenna = 'im_c.png'
#binarize_image(orig_Lenna, bin_Lenna, 127)

#load images from files
pattern = 'im_c.png'
test = 'im_c.png'
face = Image.open(pattern)#plt.imread('im_c.png')
template = Image.open(test)#plt.imread('im_blank.png')

#треба так робити. бо картинка може бути кольорова
binarize_image(test, test, 127)
binarize_image(pattern, pattern, 127)

face = Image.open(pattern)
template = Image.open(test)

#convert to numpy matrix
face = np.matrix(face) 
template = np.matrix(template)

face = np.matrix([[0, 0, 0],
                  [0, 255, 0],
                  [0, 0, 0],
                  [0, 255, 0],
                  [0, 255, 0]])
template = np.matrix([[0, 0, 0],
                  [0, 255, 0],
                  [0, 0, 0],
                  [0, 255, 0],
                  [0, 255, 0]])

print('Writing images...')
with open('A.txt', 'w') as f:
    for line in face:
        np.savetxt(f, line, fmt='%d')
matr = []
temp_row = []
print('reading from file...')
with open('A.txt') as f:
    for line in f:
        matr.append(line)

#matr = [int(item) for item in matr]
matr = [item.rstrip().split(' ')for item in matr]

matr = np.matrix(matr)
print(matr)
    
'
#зробити дві А поки що без пробілів
temp_row = []
for row in face:
    temp_row += 2 * [row]
face = np.reshape(temp_row, (face.shape[0], face.shape[1] * 2))
print('binarized template type: ')
print(template.shape)
print('face type: ')
print(face.shape)

#запис зображдень у тестовий файл
print('Writing images...')
with open('facefile.txt', 'w') as f:
    for line in face:
        np.savetxt(f, line, fmt='%.2f')

with open('templatefile.txt', 'w') as f:
    for line in template:
        np.savetxt(f, line, fmt='%.2f')
print('Done!')

#обчислення кореляції. може і не треба буде
print('Starting calculations...')
corr = signal.correlate2d(face, template, 'same')
print('corr type: ')
print(corr.shape)
y, x = np.unravel_index(np.argmax(corr), corr.shape)
print('Calculations finihed')

corr = np.matrix(corr)
with open('corrfile.txt', 'w') as f:
    for line in corr:
        np.savetxt(f, line, fmt='%.2f')
        
im_list = [face, template, corr]
show_images(im_list)


'''
fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1, figsize=(6, 15))

ax_orig.imshow(face, cmap='gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()

ax_template.imshow(template, cmap='gray')
ax_template.set_title('Template')
ax_template.set_axis_off()

ax_corr.imshow(corr, cmap='gray')
ax_corr.set_title('Cross-correlation')
ax_corr.set_axis_off()
ax_orig.plot(x, y, 'ro')
fig.show()

'''
