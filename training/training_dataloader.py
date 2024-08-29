from torch.utils.data import Dataset, IterableDataset
import torch
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import rasterio
import os
import glob
import warnings
import random
import numpy as np
# Suppress runtime warnings # HFDT-CONDA
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
from datetime import datetime
import json

def norm(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))

def satelliteimage(inputpath):
    color=[]
    with rasterio.open(inputpath, 'r') as src:
        # print(src.shape)
        S_images=src.read()
        b2 = norm(S_images[0].astype(np.float32))
        b3 = norm(S_images[1].astype(np.float32))
        b4 = norm(S_images[2].astype(np.float32))

        rgb = np.dstack((b4,b3,b2))
        return rgb


def satelliteimage_info(inputpath):
    color=[]
    with rasterio.open(inputpath, 'r') as src:
        # print(src.shape)
        S_images=src.read()
        b2 = norm(S_images[0].astype(np.float32))
        b3 = norm(S_images[1].astype(np.float32))
        b4 = norm(S_images[2].astype(np.float32))

        # Create RGB
        rgb = np.dstack((b4,b3,b2))

        # Visualize RGB

        # plt.imshow(rgb)
        # plt.axis('off')

        h=src.height
        w=src.width 

        Image=os.path.basename(inputpath)
        count=0
        for height in range(h):
            for width in range(w):
                r,g,b,nir,re,db=src.read(1)[height,width],src.read(2)[height,width],src.read(3)[height,width],src.read(4)[height,width],src.read(5)[height,width],src.read(6)[height,width]

                GLI=(2*g-r-b)/(2*g+r+b)

                # GRVIupper=g.astype(float)-r.astype(float)
                # GRVIlower=g.astype(float)+r.astype(float)
                # GRVI=(GRVIupper.astype(float)/GRVIlower.astype(float))

                NGRDIupper=r.astype(float)-g.astype(float)
                NGRDIlower=g.astype(float)+r.astype(float)
                NGRDI=(NGRDIupper.astype(float)/NGRDIlower.astype(float))

                NDVIupper=nir.astype(float)-r.astype(float)
                NDVIlower=nir.astype(float)+r.astype(float)
                NDVI=(NDVIupper.astype(float)/NDVIlower.astype(float))

                GNDVIupper=nir.astype(float)-g.astype(float)
                GNDVIlower=nir.astype(float)+g.astype(float)
                GNDVI=(GNDVIupper.astype(float)/GNDVIlower.astype(float))

                SAVIupper=1.5*(nir.astype(float)-r.astype(float))
                SAVIlower=nir.astype(float)+r.astype(float)+0.5
                SAVI=(SAVIupper.astype(float)/SAVIlower.astype(float))

                NDREupper=nir.astype(float)-re.astype(float)
                NDRElower=nir.astype(float)+re.astype(float)
                NDRE=(NDREupper.astype(float)/NDRElower.astype(float))

                color.append(
                    {'Imagename':Image,
                     'Red':r,
                     'Green':g,
                     'Blue':b,
                     'GLI':GLI,
                     # 'GRVI':GRVI,
                     'NGRDI':NGRDI,
                     'NDVI':NDVI,
                     'GNDVI':GNDVI,
                     'SAVI':SAVI,
                     'NDRE':NDRE,
                     'RedEdge':re,
                     'DeepBlue':db,
                     'Nir':nir
                    } 
                )
    band_values=pd.DataFrame(color)
    band_values=band_values[band_values['Red']>0] ##Remove the pixel values that is equal to zero
    band_values=band_values.groupby('Imagename')[list(band_values.describe().columns)].agg(['mean', 'median', 'sum']).reset_index()
    band_values.columns = band_values.columns.map('_'.join)
    
    return round_floats_in_dict(band_values.loc[0].to_dict()), rgb

def round_floats_in_dict(sample):
    def round_float(value):
        if isinstance(value, float):
            return round(value, 3)
        else:
            return value

    rounded_sample = {key: round_float(value) for key, value in sample.items()}
    return rounded_sample


def pad_image(image, target_shape):
    target_c, target_h, target_w = target_shape
    resized_image = cv.resize(image, (target_w, target_h), interpolation=cv.INTER_LINEAR)
    reshaped_image = np.transpose(resized_image, (2, 0, 1))
    return reshaped_image

def extend_batch_dimension(sample):
    extended_sample = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            extended_sample[key] = value.unsqueeze(0)
        elif isinstance(value, str):
            extended_sample[key] = [value]
        else:
            extended_sample[key] = value 
    return extended_sample



class CustomDataset(Dataset):
    def __init__(self, data_path=".././2023/DataPublication_final/GroundTruth/train_HIPS_HYBRIDS_2023_V2.3.csv", 
                 date_path=".././2023/DataPublication_final/GroundTruth/DateofCollection.xlsx",
                 mother_path=".././2023/DataPublication_final/",
                 max_seq_len=10,
                 is_testing=False,
                 device='cpu'):
        self.data_path = data_path
        self.date_path = date_path
        self.data = pd.read_csv(data_path)
        if not is_testing:
            self.data = self.data.dropna(subset=['yieldPerAcre']).reset_index(drop=True)
        self.date = pd.read_excel(date_path)

        self.temp_locations = self.data['location'].unique()


        path_to_files = os.listdir(mother_path)
        self.paths_to_image_files = [os.path.join(mother_path, x) for x in path_to_files if x in ['Satellite', 'UAV']]
        self.image_path_location = [os.listdir(x) for x in self.paths_to_image_files]
        
        self.max_seq_len = max_seq_len
        self.sat_image_size = (3,15,25)
        # self.image_size = (3,15,25)
        self.uav_image_size = (3,400,750)
        self.max_loc_len = 5
        #
        self.update_image_location()
        #
        self.device = device

    def __len__(self):
        return len(self.data)

    def update_image_location(self):
        # add 3 new columns: final_paths, image_types, dates_
        self.data['final_image_paths'] = None
        self.data['image_types'] = None
        self.data['image_dates_'] = None
        for idx in range(len(self.data)):
            location = self.data.loc[idx]['location']
            rangeno = self.data.loc[idx]['range']
            row = self.data.loc[idx]['row']
            experiment = self.data.loc[idx]['experiment']
            final_paths = []
            image_types=[]
            dates_ = []
            for files in self.image_path_location[0]:
                if not files == location:
                    continue
                finallocation = location
                finalimagefolder = [os.path.join(x, finallocation) for x in self.paths_to_image_files]

            for locationfolder in finalimagefolder:
                timepointfolder = sorted(os.listdir(locationfolder))
                imagetype = locationfolder.split('/')[-2]
                location = locationfolder.split('/')[-1]            
                timepointpath = [os.path.join(locationfolder, x) for x in timepointfolder]

                for timepointpath_ in timepointpath:
                    imagefiles = os.listdir(timepointpath_)

                    for images in imagefiles:
                        range_ = images.split('_')[1]
                        row_ = images.split('_')[2].split('.')[0]
                        experiment_ = images.split('_')[0].split('-')[2]

                        if str(range_) == str(rangeno) and str(row_) == str(row) and str(experiment) == str(experiment_):
                            timepoint = images.split('_')[0].split('-')[1]
                            finalpath = os.path.join(timepointpath_, images)
                            final_paths.append(finalpath)
                            image_types.append(imagetype)
                            dates_.append(self.date.loc[(self.date['Location'] == location) & 
                                                  (self.date['Image'] == imagetype) & 
                                                  (self.date['time'] == timepoint)]['Date'].to_string(index=False))

            self.data.at[idx, 'final_image_paths'] = final_paths
            self.data.at[idx, 'image_types'] = image_types
            self.data.at[idx, 'image_dates_'] = dates_

    def __getitem__(self, idx):
        # Location
        location = self.data.loc[idx]['location']
        # Irrigation
        irrigationProvided = self.data.loc[idx]['irrigationProvided']
        nitrogenTreatment = self.data.loc[idx]['nitrogenTreatment']
        poundsOfNitrogenPerAcre = self.data.loc[idx]['poundsOfNitrogenPerAcre']
        # Plot Geometry
        plotLength = self.data.loc[idx]['plotLength']
        row = self.data.loc[idx]['row']
        rangeno = self.data.loc[idx]['range']
        experiment = self.data.loc[idx]['experiment']
        plotno = self.data.loc[idx]['plotNumber']
        # Plant Genotype
        genotype = self.data.loc[idx]['genotype']
        # Planting Date
        plantingDate = self.data.loc[idx]['plantingDate']
        # Outputs
        totalStandCount = self.data.loc[idx]['totalStandCount']
        daysToAnthesis = self.data.loc[idx]['daysToAnthesis']
        GDDToAnthesis = self.data.loc[idx]['GDDToAnthesis']
        yieldPerAcre = self.data.loc[idx]['yieldPerAcre']
        
        final_paths = self.data.loc[idx]['final_image_paths']
        image_types = self.data.loc[idx]['image_types']
        dates_ = self.data.loc[idx]['image_dates_']
        satelliteimagelist = []
        satelliteimageinfo = []
        satelliteimagedate = []
        for an_image, this_image_type, image_cap_date in zip(final_paths, image_types, dates_):
            if this_image_type=='Satellite':
                this_image_info, rgb_image = satelliteimage_info(an_image)
                # this_image_info["date captured"] = image_cap_date
                reshaped_image =  pad_image(rgb_image, self.sat_image_size)
                satelliteimageinfo.append(normalize_dict_values(this_image_info))
                satelliteimagedate.append(str(image_cap_date))
                satelliteimagelist.append(reshaped_image)
            else:
                pass # uav images, later
        
        satellite_images_array = np.zeros((self.max_seq_len, *self.sat_image_size), dtype=np.float32)
        satellite_images_info_array = np.zeros((self.max_seq_len, 36), dtype=np.float32)
        for i in range(len(satelliteimagelist)):
            satellite_images_array[i] = satelliteimagelist[i]
            satellite_images_info_array[i] = satelliteimageinfo[i]

        sample = {
            'location':location,
            'irrigation provided': str(irrigationProvided),
            'nitrogen treatment': nitrogenTreatment,
            'pounds of nitrogen per acre': str(poundsOfNitrogenPerAcre),
            'plot length': str(plotLength),
            'genotype': str(genotype),
            'planting date': str(plantingDate),
            'image number': str(len(satelliteimagelist)),
            'image dates': str(satelliteimagedate),
            'row': str(row),
            'range': str(rangeno),
            'experiment': str(experiment),
            'plot number': str(plotno),
        }

        y = (yieldPerAcre-150)/100

        info = {
            'totalStandCount': totalStandCount,
            'daysToAnthesis': daysToAnthesis,
            'GDDToAnthesis': GDDToAnthesis,
            'yieldPerAcre': yieldPerAcre,
            'satelliteImages': satellite_images_array,
            'satelliteImagesLen': len(satelliteimagelist),
            'satelliteImagesInfo': satellite_images_info_array
        }
        return json.dumps(sample), y, info



class IterableCustomDataset(IterableDataset):
    def __init__(self, custom_dataset):
        self.custom_dataset = custom_dataset
    
    def __iter__(self):
        while True:
            idx = np.random.choice(len(self.custom_dataset))
            yield self.custom_dataset[idx]



def normalize_dict_values(data_dict):
    keys_order = [
        'Red_mean', 'Red_median', 'Red_sum',
        'Green_mean', 'Green_median', 'Green_sum',
        'Blue_mean', 'Blue_median', 'Blue_sum',
        'GLI_mean', 'GLI_median', 'GLI_sum',
        'NGRDI_mean', 'NGRDI_median', 'NGRDI_sum',
        'NDVI_mean', 'NDVI_median', 'NDVI_sum',
        'GNDVI_mean', 'GNDVI_median', 'GNDVI_sum',
        'SAVI_mean', 'SAVI_median', 'SAVI_sum',
        'NDRE_mean', 'NDRE_median', 'NDRE_sum',
        'RedEdge_mean', 'RedEdge_median', 'RedEdge_sum',
        'DeepBlue_mean', 'DeepBlue_median', 'DeepBlue_sum',
        'Nir_mean', 'Nir_median', 'Nir_sum'
    ]
    
    normalized_values = []
    for key in keys_order:
        value = data_dict.get(key)
        if value is None:
            normalized_values.append(0)
        else:
            if key in ['GLI_mean', 'GLI_median']:
                normalized_values.append(value * 10)
            elif key == 'GLI_sum':
                normalized_values.append(value / 10)
            elif key == 'NGRDI_sum':
                normalized_values.append(value / 100)
            elif key == 'NDVI_sum':
                normalized_values.append(value / 100)
            elif key == 'GNDVI_sum':
                normalized_values.append(value / 100)
            elif key == 'SAVI_sum':
                normalized_values.append(value / 100)
            elif key == 'NDRE_sum':
                normalized_values.append(value / 100)
            elif key in ['NGRDI_mean', 'NGRDI_median']:
                normalized_values.append(value)
            elif key in ['NDVI_mean', 'NDVI_median']:
                normalized_values.append(value)
            elif key in ['GNDVI_mean', 'GNDVI_median']:
                normalized_values.append(value)
            elif key in ['SAVI_mean', 'SAVI_median']:
                normalized_values.append(value)
            elif key in ['NDRE_mean', 'NDRE_median']:
                normalized_values.append(value)
            elif 'mean' in key or 'median' in key:
                normalized_values.append(value / 1000)
            elif 'sum' in key:
                normalized_values.append(value / 1e5)
            else:
                normalized_values.append(value)
                
    return normalized_values