from pyexiv2 import Image
import numpy as np

def get_image_taking_conditions(image_dir):
    info = Image(image_dir)
    exif_info = info.read_exif()
    xmp_info = info.read_xmp()
    re = dict()
    re['shutter'] = exif_info['Exif.Photo.ExposureTime']
    re['ISO'] = exif_info['Exif.Photo.ISOSpeedRatings']
    re['aperture'] = exif_info['Exif.Photo.MaxApertureValue']
    re['image_name'] = image_dir.split('/')[-1]
    re['altitude'] = float(xmp_info['Xmp.drone-dji.RelativeAltitude'][1:])
    #print (image_name,xmp_info['Xmp.drone-dji.RelativeAltitude'])
    return re


def py_cpu_nms(dets,thresh=0.25):  
    """
    Pure Python NMS baseline.
    Added extra step to reduce small box inside large box
    """ 
    dets = np.asarray(dets) 
    x1 = np.asarray([float(i) for i in dets[:, 0]]) 
    y1 = np.asarray([float(i) for i in dets[:, 1]]) 
    x2 = np.asarray([float(i) for i in dets[:, 2]])   
    y2 = np.asarray([float(i) for i in dets[:, 3]])  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    order = areas.argsort()[::-1]  
    keep = []  
    while order.size > 0:  
        i = order[0]  
        keep.append(i)  
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        ovr = 1.*inter / (areas[i] + areas[order[1:]] - inter)  
        inds = np.where((ovr <= thresh) & (areas[order[1:]]>1.2*inter))[0]  
        order = order[inds + 1]  
    return keep


def get_sub_image(mega_image,overlap=0.2,ratio=1,cropSize = 512):
	#mage_image: original image
	#ratio: ratio * 512 counter the different heights of image taken
	#return: list of sub image and list fo the upper left corner of sub image
	if (mega_image.shape[0] ==cropSize):
		return [mega_image],[[0,0]]
	coor_list = []
	sub_image_list = []
	w,h,c = mega_image.shape
	size  = int(ratio*cropSize)
	num_rows = int(w/int(size*(1-overlap)))
	num_cols = int(h/int(size*(1-overlap)))
	new_size = int(size*(1-overlap))
	for i in range(num_rows+1):
		if (i == num_rows):
			for j in range(num_cols+1):
				if (j==num_cols):
					sub_image = mega_image[-size:,-size:,:]
					coor_list.append([w-size,h-size])
					sub_image_list.append (sub_image)
				else:
					sub_image = mega_image[-size:,new_size*j:new_size*j+size,:]
					coor_list.append([w-size,new_size*j])
					sub_image_list.append (sub_image)
		else:
			for j in range(num_cols+1):
				if (j==num_cols):
					sub_image = mega_image[new_size*i:new_size*i+size,-size:,:]
					coor_list.append([new_size*i,h-size])
					sub_image_list.append (sub_image)
				else:
					sub_image = mega_image[new_size*i:new_size*i+size,new_size*j:new_size*j+size,:]
					coor_list.append([new_size*i,new_size*j])
					sub_image_list.append (sub_image)
	return sub_image_list,coor_list


def get_GSD(anno_data, camera_type='Pro2', ref_altitude=60):
    height = anno_data['altitude']
    if (camera_type == 'Pro2'):
        ref_GSD = (13.2 * ref_altitude)/(10.26*5472)
        GSD = (13.2 * height)/(10.26*5472)
    elif (camera_type == 'Air2'):
        ref_GSD = (6.4*ref_altitude)/(4.3*8000)
        GSD = (6.4*height)/(4.3*8000)
    else:
        ref_GSD = (13.2 * ref_altitude)/(10.26*5472)
        GSD = (13.2 * height)/(10.26*5472)
    return GSD, ref_GSD