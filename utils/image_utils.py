from collections import defaultdict
import numpy as np
import os
import openslide
from SlideRunner.dataAccess.annotations import *




def create_list_by_type_one_image(single_img_annotations):
    annotation_list_by_type = defaultdict(list)

    for k in single_img_annotations.keys():
        annot_class = single_img_annotations[k].agreedClass
        annotation_list_by_type[annot_class].append(k)
        
    len(annotation_list_by_type[1]), len(annotation_list_by_type[2])

    return annotation_list_by_type

def get_slides(DB):
    """
    Get a list of tuples (slide_num, WSI_name) from the database
    """

    getslides = """SELECT uid, filename FROM Slides"""
    return DB.execute(getslides).fetchall()

def calculate_bbox(x, y, r=25, m=1):
    """
    Calculates bounding box of an annotation
    Inputs:
    x: annotation center X coordinate
    y: annotation center Y coordinate
    r: Radius around the annotation center
    m: Microscope mag
    
    Output:
    a list of 4 numbers that defines the bounding box
    """
    d = 2 * r / m
    
    x_min = (x - r) / m
    y_min = (y - r) / m
    
    x_max = x_min + d
    y_max = y_min + d
    
    bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
    
    return bbox


def create_patch_center_identifier(filename, nx, ny):

    return f"{filename}_{str(nx)}_{str(ny)}.jpg"

def calculate_patch_count_in_WSI(wsi_x, wsi_y, patch_size):
    """
    Calculates the number of patches in a WSI, 
    
    Inputs:
    wsi_x: number of pixels in X
    wsi_y: number of pixels in Y
    patch_size: size of each patch
    
    Returns:
    number of patches in the WSI. 
    """
    nx = int(wsi_x / patch_size)
    ny = int(wsi_y / patch_size)

    return nx*ny

def find_patch_index_within_WSI(x, y, patch_size):
    """
    Calculates the index of the patch where the annotation belongs

    Inputs:
    x: annotation center X coordinate
    y: annotation center X coordinate
    patch_size: size of each patch

    Returns: (nx, ny): index of the patch in units of patch size where the annotation is present
    """
    nx = int(x / patch_size)
    ny = int(y / patch_size)

    return (nx, ny)

def get_patch_annotations(patch_size, image_size, single_img_annotations):
    """
    Create a dict where key are tuples of patch center coordinates and
    values are list of bounding boxes of annotations that are within that patch

    Inputs:
    patch_size: size of the patch
    image_size: tuple (x,y) that denotes image size
    single_img_annotations: annotations for a single image
    """
    cls_2_count = 0

    # Sample 20 random mitotic cell regions

    annotation_list_by_type = create_list_by_type_one_image(single_img_annotations)

    sample_of_mitotic_cells = np.random.choice(annotation_list_by_type[2], 10)
    print(f"Sample of annotations:  {sample_of_mitotic_cells}")

    img_x, img_y = image_size
    patch_annotation_list = defaultdict(list)

    #for id, annotation in database.annotations.items():
    # Loop through all the annotations in the imagw
    for annot_uid in sample_of_mitotic_cells: #database.annotations.items():
    
        annotation = single_img_annotations[annot_uid]
        if annotation.deleted or annotation.annotationType != AnnotationType.SPOT:
            continue
        else:
            #get X and Y coordinates of the annotation
            x = annotation.x1 
            y = annotation.y1
            agreed_classs = annotation.agreedClass
            annot_uid = annotation.uid #unique ID of the annotation
            #print(annotation.annotationType, annotation.agreedClass, x, y)
            bbox = calculate_bbox(x,y)

            #Calculate the center of the patch where this annotation is present
            n_x = int(x/patch_size)
            n_y = int(y/patch_size)

            patch_center_x = int(n_x*patch_size + patch_size/2)
            patch_center_y = int(n_y*patch_size + patch_size/2)

            patch_annotation_list[(patch_center_x, patch_center_y)].append([agreed_classs, bbox])
            #Assign to the list of annotations for this patch
            #print(bbox)
    
    return patch_annotation_list
            

def create_image_patches(slide_dir, DB, patch_size):
    
    slide_list = get_slides(DB) #list of all slides in the database

    global_patch_count = 0
    coco_image_labels = []
    for n in range(len(slide_list)):
        slide_num = n
        currslide, filename = slide_list[slide_num]
        #print(f"\n\nSlide number: {currslide},  Name: {filename}\n")

        #load the WSI into memory
        #database.loadIntoMemory(currslide)
        
        #img_annotaton = database.annotations
        #img_annotation_list = create_list_by_type_one_image(img_annotaton)
        
        slide_path = os.path.join(slide_dir, filename)  #basepath + os.sep + filename
        slide = openslide.open_slide(str(slide_path))
        
        dx, dy = slide.dimensions
        print(f"\n\nSlide number: {currslide},  Name: {filename}, Slide dimension: {slide.dimensions}\n")

        for i in range(int(dx/patch_size)):
            for j in range(int(dy/patch_size)):
                global_patch_count += 1
                image_dict = {}
                #patch_name = f"{filename}_{str(i)}_{str(j)}_{str(global_patch_count)}"
                patch_name = f"{filename}_{str(i)}_{str(j)}"
                image_dict['license']=1
                image_dict['file_name'] =  f'{patch_name}.jpg',
                image_dict['coco_url'] =  'http://images.cocodataset.org/val2017/000000397133.jpg',
                image_dict['height'] = patch_size
                image_dict['width'] =  patch_size
                image_dict['date_captured'] = '2013-11-14 17:02:52',
                image_dict['flickr_url']= 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg'
                image_dict['id']= global_patch_count

                coco_image_labels.append(image_dict)
                if global_patch_count%1000 ==0:    
                    print(global_patch_count, "  ***  ", image_dict)

    return coco_image_labels