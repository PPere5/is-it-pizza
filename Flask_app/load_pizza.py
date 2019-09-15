def verify_image(img_file):
    from skimage import io
    try:
        img = io.imread(img_file)
    except:
        return False
    return True


def load_pizza_data(img_rows, img_cols, test_ratio, seed, n_per_folder):


    import os
    import numpy as np

    import cv2

    import imageio

    from sklearn.model_selection import train_test_split


    #Defining the File Path

    filepath="D:/Projects deposit/Is_it_pizza/"

    pizza_folders = [
        'Pizza',
        'Pizza slice',
        'American Pizza',
        'Italian Pizza',
        'World Pizza'
        'Traditional Pizza',
        'Fast food pizza'
        'hot pizza',
        'vegan pizza',
        'meat pizza'
    ]

    not_pizza_folders = [
        'Person',
        'people chatting in front of webcam',
        'tools',
        'food -pizza',
        'round objects -pizza',
        'red and yellow object -pizza',
        'red and yellow round -pizza'
        'colourful furniture',
        'wheels',
        'face'
    ]

    if n_per_folder != 'all':
        n_pizza, n_not = n_per_folder, n_per_folder

    #Loading the Images

    images = []
    label = []

    image_extensions = ['.png', '.jpg']

    for fold in pizza_folders:
        path = filepath + fold + '/'
        pizza = os.listdir(path)
        if n_per_folder != 'all':
            pizza = pizza[0:n_pizza+1]

        for i in pizza:
            filename, file_extension = os.path.splitext(path+i)
            if file_extension in image_extensions:

                if verify_image(path + i):
                    image = np.array(imageio.imread(path + i))

                    if image.ndim == 3:
                        if image.shape[2] >= 3:

                            #deleting the alpha channel
                            if image.shape[2] > 3:
                                image = image[:, :, 0:3]

                            images.append(image)
                            label.append(0) #for pizza images

    for fold in not_pizza_folders:
        path = filepath + fold + '/'
        not_pizza = os.listdir(path)
        if n_per_folder != 'all':
            not_pizza = not_pizza[0:n_not+1]

        for i in not_pizza:
            filename, file_extension = os.path.splitext(path + i)
            if file_extension in image_extensions:

                if verify_image(path + i):
                    image = imageio.imread(path + i)

                    if image.ndim == 3:
                        if image.shape[2] >= 3:

                            # deleting the alpha channel
                            if image.shape[2] > 3:
                                image = image[:, :, 0:3]

                            images.append(image)
                            label.append(1)  # for not pizza images

    #resizing all the images

    len_images = len(images)

    for i in range(0, len_images):
        images[i] = cv2.resize(images[i], (img_rows, img_cols))

    #converting images to arrays

    images = np.array(images)
    label = np.array(label)

    # print(images.shape)
    # print(images[3].shape)
    # print(images[10].shape)



    x_train, x_test, y_train, y_test = train_test_split(images, label, test_size=test_ratio, random_state=seed)

    # print(x_test.shape)
    # for x in x_test:
    #     print(x.shape)
    # print(x_train.shape)
    #
    # for x in x_train:
    #     print(x.shape)
    #
    # print(y_test.shape)
    # print(y_test[0].shape)
    # print(y_train.shape)
    # print(y_train[0].shape)

    return (x_train, y_train), (x_test, y_test)

