# importing google_images_download module
from google_images_download import google_images_download

# creating object
response = google_images_download.googleimagesdownload()

search_queries_pizza = [
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

search_queries_not_pizza = [
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

def downloadimages(query, lim, word):
    # keywords is the search query
    # format is the image file format
    # limit is the number of images to be downloaded
    # print urs is to print the image file url
    # size is the image size which can
    # be specified manually ("large, medium, icon")
    # aspect ratio denotes the height width ratio
    # of images to download. ("tall, square, wide, panoramic")
    arguments = {"keywords": query,
                 "limit": lim,
                 "prefix": word,
                 "chromedriver": "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
                 "print_urls": True,
                 "aspect_ratio": "panoramic",
                 "output_directory": "D:/Projects deposit/Is_it_pizza"
                 }
    try:
        response.download(arguments)

    # Handling File NotFound Error
    except FileNotFoundError:
        arguments = {"keywords": query,
                     "chromedriver": "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
                     "limit": lim,
                     "prefix": word,
                     "print_urls": True}

        # Providing arguments for the searched query
        try:
            # Downloading the photos based
            # on the given arguments
            response.download(arguments)
        except:
            pass


# Driver Code
for query in search_queries_pizza:
    downloadimages(query, 100, 'Pizza')
    print()

for query in search_queries_not_pizza:
    downloadimages(query, 100, 'Not_Pizza')
    print()