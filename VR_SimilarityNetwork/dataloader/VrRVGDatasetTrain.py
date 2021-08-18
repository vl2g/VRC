import os
import numpy as np
from torch.utils.data import Dataset
import glob
import xml.etree.ElementTree as ET
from collections import defaultdict
import glob

import random
from tqdm import tqdm
import torch

from utils.sampling_utils import get_iou
from utils.utils import read_json


class VrRVGDatasetTrain(Dataset):
    def __init__(self, data_config, vrNetwork):
        self.data_config = data_config
        
        self.vrNetwork = vrNetwork
        self.vrNetwork = self.vrNetwork.cuda()
        if(self.data_config.VREncoderEmbeddings == "VTransE"):
            chkpt=torch.load(self.data_config.VREncoder_Net_Checkpoint) 
            self.vrNetwork.load_state_dict(chkpt["model_state_dict"])
        self.vrNetwork.eval()

        self.xml_path = self.data_config.XML_FILE_PATH_VrRVG
        self.npy_file_path = self.data_config.NPY_FILE_PATH

        self.all_xml_files_path=glob.glob(self.xml_path+"/*.xml")
        self.predicates_name_to_id=read_json(data_config.train_predicates_path)
        self.relation_id_to_images=defaultdict(list)
        self.get_relation_id_to_images()
        
        self.sampled_bags=self.load_samples() 
        self.bag_features=self.sample_generator() # all vtranse features of relations in images in a bag

    def extract_vtranse_embedding_from_relation(self,info,feat,subjs,objs,image_width,image_height):
        relation={}
        bnd_boxx=info.item().get('bbox')[subjs] # [xmin ymin xmax ymax]
        bnd_box=bnd_boxx.copy()
        bnd_box[0]=float(bnd_box[0]/image_width)
        bnd_box[2]/=image_width
        bnd_box[1]/=image_height
        bnd_box[3]/=image_height
        relation["sub_class_scores"] = info.item()["class_scores"][subjs]
        relation["obj_class_scores"]= info.item()["class_scores"][objs]
        relation["sub_bnd_box"]=bnd_box
        
        bnd_boxxx=info.item().get('bbox')[objs] # [xmin ymin xmax ymax]
        bnd_box=bnd_boxxx.copy()
        bnd_box[0]/=image_width
        bnd_box[2]/=image_width
        bnd_box[1]/=image_height
        bnd_box[3]/=image_height
        relation["obj_bnd_box"]=bnd_box
        
        relation["sub_roi_features"]=feat[subjs]
        relation["obj_roi_features"]=feat[objs]
        if(self.data_config.VREncoderEmbeddings == "VTransE"):
            embedding = self.get_vtranse_embedding(relation)
        else:
            embedding = self.get_concat_embeddings(relation)

        return embedding

    def get_vtranse_embedding(self,sample_batched):

        with torch.no_grad():

            subj_sp=torch.tensor(sample_batched["sub_bnd_box"]).float().cuda()
            obj_sp=torch.tensor(sample_batched["obj_bnd_box"]).float().cuda()
            
            subj_cls=torch.tensor(sample_batched["sub_class_scores"]).float().cuda()
            obj_cls=torch.tensor(sample_batched["obj_class_scores"]).float().cuda()
            
            sub_feat=torch.tensor(sample_batched["sub_roi_features"]).float().cuda()
            obj_feat=torch.tensor(sample_batched["obj_roi_features"]).float().cuda()

            _, vr_emb = self.vrNetwork.forward_inference(subj_sp,subj_cls,sub_feat,obj_sp,obj_cls,obj_feat)
            vr_emb_numpy=vr_emb.cpu().detach().numpy()
        return vr_emb_numpy
        
    def get_concat_embeddings(self, sample_batched):
        with torch.no_grad():
            subj_sp=torch.tensor(sample_batched["sub_bnd_box"]).float()
            obj_sp=torch.tensor(sample_batched["obj_bnd_box"]).float()
            
            subj_cls=torch.tensor(sample_batched["sub_class_scores"]).float()
            obj_cls=torch.tensor(sample_batched["obj_class_scores"]).float()
            
            sub_feat=torch.tensor(sample_batched["sub_roi_features"]).float()
            obj_feat=torch.tensor(sample_batched["obj_roi_features"]).float()

            sub_emb = torch.cat([subj_sp, subj_cls, sub_feat])
            ob_emb = torch.cat([obj_sp, obj_cls, obj_feat])

            vr_emb = torch.cat([sub_emb, ob_emb])

            vr_emb = vr_emb.cpu().detach().numpy()
        return vr_emb
        
    def sample_generator(self):
        bags=[]
        i=0
        for  bag in tqdm(self.sampled_bags, desc="generating dataset"): # what is self.sampled_bags?? a list which has dicts {"relation_id" :relation_id, "images_ids" : img_ids of bag size} 
            i+=1
            bag_relation_id=bag["relation_id"] # relation id is of common relation of the bag
            relations_per_bag={}
            relations_per_bag["relation_id"]=bag_relation_id
            relations_per_bag["relations"]=[] # n1,n2,n3 relations in images
            bag_size = len(bag["images_ids"])
            try:
                for img in bag["images_ids"]:
                    relations_per_image={}
                    img_name=img.split(".")[0]

                    relations_per_image["image_name"]=img_name
                    relations_per_image["positive_relations"]=[]
                    relations_per_image["negative_relations"]=[]

                    npy_info_name=img_name+"_info.npy"
                    npy_feat_name=img_name+".npy"
                    info=np.load(os.path.join(self.npy_file_path,npy_info_name),allow_pickle=True)
                    feat=np.load(os.path.join(self.npy_file_path,npy_feat_name),allow_pickle=True)
                    xml_file_path=self.xml_path+"/"+img_name+".xml"

                    xml_data = ET.parse(xml_file_path)
                    root = xml_data.getroot()
                    positive_relations_id_from_xml=[]
                    all_relations_in_xml = root.findall('./relation')
                    
                    # function of this for loop : finding positive relations in xml
                    for rel in all_relations_in_xml:
                        predicate=str(rel.find("predicate").text)
                        try:
                            verify_test_train = self.predicates_name_to_id[predicate]
                            if(verify_test_train == bag_relation_id):
                                rel_sub_id=str(rel.find('./subject_id').text)
                                rel_obj_id=str(rel.find('./object_id').text)
                                relation_key=rel_sub_id+rel_obj_id
                                positive_relations_id_from_xml.append(relation_key)
                        except:
                            # if rel is of different test/train set : basically ignore this code
                            pass
                          
                    ## refactored till here
                    for rel in all_relations_in_xml:
                        predicate=str(rel.find("predicate").text)
                        try:
                            check=(self.predicates_name_to_id[predicate])
                            if(check==bag_relation_id):
                                rel_sub_id=int(rel.find('./subject_id').text)
                                rel_obj_id=int(rel.find('./object_id').text)
                                current_rel_key=str(rel_sub_id)+str(rel_obj_id)
                                for sub in root.findall('./object'): 
                                    for obj in root.findall('./object'):
                                        flag=0
                                        sub_id=int(sub.find('object_id').text)
                                        obj_id=int(obj.find('object_id').text)
                                        if(rel_sub_id==sub_id and rel_obj_id==obj_id):
                                            subject_bbox={}
                                            object_bbox={}

                                            
                                            subject_bbox["xmin"]=float(sub.find('bndbox').find('xmin').text)
                                            subject_bbox["xmax"]=float(sub.find('bndbox').find('xmax').text)
                                            subject_bbox["ymin"]=float(sub.find('bndbox').find('ymin').text)
                                            subject_bbox["ymax"]=float(sub.find('bndbox').find('ymax').text)

                                            object_bbox["xmin"]=float(obj.find('bndbox').find('xmin').text)
                                            object_bbox["xmax"]=float(obj.find('bndbox').find('xmax').text)
                                            object_bbox["ymin"]=float(obj.find('bndbox').find('ymin').text)
                                            object_bbox["ymax"]=float(obj.find('bndbox').find('ymax').text)

                                            #get subjects and objects roi index from npy file where the iou is greater than the threshold for subjects and objects
                                            # ERROR in this Function : now fixed
                                            subject_roi_index, neutral_sub_roi_index = self.get_subject_roi_index(root, subject_bbox, info, bag_relation_id, positive_relations_id_from_xml, all_relations_in_xml, type="subject")
                                            object_roi_index, neutral_obj_roi_index = self.get_subject_roi_index(root, object_bbox, info, bag_relation_id, positive_relations_id_from_xml, all_relations_in_xml, type="object")

                                            image_width = int(info.item().get('image_width'))
                                            image_height = int(info.item().get('image_height'))
                                            sampling=0
                                            # extract positive relations embeddings
                                            for i,subjs in enumerate(subject_roi_index):
                                                for j,objs in enumerate(object_roi_index):
                                                    embedding = self.extract_vtranse_embedding_from_relation(info,feat,subjs,objs,image_width,image_height)
                                                    # print("appending positive-positive pairs relations in img")
                                                    relations_per_image["positive_relations"].append(embedding) 
                                                    sampling+=1

                                            rois_info=info.item().get('bbox')
                                            rois=rois_info.shape[0]
                                            cntt=0
                                            objs_len=len(object_roi_index)

                                            # taking positive-negative/ negative-positive pairs as negative samples
                                            for i,subjs in enumerate(subject_roi_index):
                                                for j in range(0,objs_len):
                                                    objs=random.randint(0,rois-1)
                                                    if(objs not in object_roi_index):
                                                        embedding=self.extract_vtranse_embedding_from_relation(info,feat,subjs,objs,image_width,image_height)
                                                        # print("appending positive-negative pairs relations in img")
                                                        relations_per_image["negative_relations"].append(embedding) 
                                                        sampling+=1
                                                    else:
                                                        j-=1
                                            subjs_len=len(subject_roi_index)
                                            for i,objs in enumerate(object_roi_index):
                                                for j in range(0,subjs_len):
                                                    subjs=random.randint(0,rois-1)
                                                    if(subjs not in subject_roi_index):
                                                        embedding=self.extract_vtranse_embedding_from_relation(info,feat,subjs,objs,image_width,image_height)
                                                        # print("appending positive-negative pairs relations in img")

                                                        relations_per_image["negative_relations"].append(embedding) 
                                                        sampling+=1
                                                    else:
                                                        j-=1

                                        # taking negative-negative pairs 
                                            for i in range(subjs_len):
                                                subjs=random.randint(0,rois-1)
                                                if(subjs in subject_roi_index or subjs in neutral_sub_roi_index):
                                                    i-=1
                                                else:
                                                    for j in range(objs_len):
                                                        objs=random.randint(0,rois-1)
                                                        if(objs in object_roi_index or objs in neutral_obj_roi_index):
                                                            j-=1
                                                        else:
                                                            embedding=self.extract_vtranse_embedding_from_relation(info,feat,subjs,objs,image_width,image_height)
                                                            # print("appending negative-negative pairs relations in img")

                                                            relations_per_image["negative_relations"].append(embedding) 
                                                            sampling+=1

                        except Exception as e: 
                            # print("exception")
                            # print(e)
                            # print("-")
                            pass
                    
                    if(len(relations_per_image["positive_relations"])!=0 and len(relations_per_image["negative_relations"])!=0):
                        # print("appending image relations in bag")
                        relations_per_bag["relations"].append(relations_per_image)
            except:
                pass
            
            if(len(relations_per_bag["relations"]) == bag_size):
                # print("appending bag in list")
                bags.append(relations_per_bag)
            else : 
                # print('len(relations_per_bag["relations"]', len(relations_per_bag["relations"]))
                pass

        return bags

    def get_subject_roi_index(self, root, bbox, info, bag_relation_id, positive_relations_id_from_xml, all_relations_in_xml, type="subject"):
        
        indexes=[]
        neutral_indexes=set()

        rois_info=info.item().get('bbox')
        rois=rois_info.shape[0]
        subj_roi_iou=[]
        # for positive rois
        for i in range(0, rois):
            bbox_roi=rois_info[i]
            bbox_dict={}
            bbox_dict["xmin"]=float(bbox_roi[0])
            bbox_dict["ymin"]=float(bbox_roi[1])
            bbox_dict["xmax"]=float(bbox_roi[2])
            bbox_dict["ymax"]=float(bbox_roi[3])

            iou=get_iou(bbox,bbox_dict)
            if(iou>0.50):
                indexes.append(i)
        #for neutral rois
        # since neutrals are ignored : all rois which are in positive are also in neutral
        for  i in range(0,rois):
            bbox_roi=rois_info[i]
            bbox_dict={}
            bbox_dict["xmin"]=float(bbox_roi[0])
            bbox_dict["ymin"]=float(bbox_roi[1])
            bbox_dict["xmax"]=float(bbox_roi[2])
            bbox_dict["ymax"]=float(bbox_roi[3])

            for rel in all_relations_in_xml:
                predicate=str(rel.find("predicate").text)
                check=(self.predicates_name_to_id[predicate])
                if(check==bag_relation_id):
                    rel_sub_id=int(rel.find('./subject_id').text)
                    rel_obj_id=int(rel.find('./object_id').text)
                    current_rel_key=str(rel_sub_id)+str(rel_obj_id)
                    if current_rel_key in positive_relations_id_from_xml:
                        for sub in root.findall('./object'):    
                            flag=0
                            xml_obj_id=int(sub.find('object_id').text)
                            if(type=="subject"):
                                if(rel_sub_id!=xml_obj_id):
                                    continue
                            elif(type=="object"):
                                if(rel_obj_id!=xml_obj_id):
                                    continue
                            subject_class=str(sub.find('name').text)
                            subject_bbox={}
                            subject_bbox["xmin"]=float(sub.find('bndbox').find('xmin').text)
                            subject_bbox["xmax"]=float(sub.find('bndbox').find('xmax').text)
                            subject_bbox["ymin"]=float(sub.find('bndbox').find('ymin').text)
                            subject_bbox["ymax"]=float(sub.find('bndbox').find('ymax').text)
                            iou=get_iou(subject_bbox,bbox_dict)
                            if(iou>0.50):
                                neutral_indexes.add(i)

        neutral_indexes=list(neutral_indexes)
        return indexes,neutral_indexes

    def load_samples(self):
        ''' returns a list of dicts. Each dict : a bag for training : {"relation_id" : id_of_predicate, "images_ids": list_of_img_ids_of_bag_size} '''
        samples=[]
        # bag_size = [4,6,8,10]
        # count_bag = np.zeros(4)
        bag_size = [2,4]
        count_bag = np.zeros(2)
        
        for relation_id,images_ids in tqdm(self.relation_id_to_images.items(), desc="returing a list of dictionary og relations"):
            images_len=len(images_ids)
            sample_per_relation = self.data_config.SAMPLE_PER_RELATION
            for _ in range(0,sample_per_relation):
                bag_n = np.argmin(count_bag)
                count_bag[bag_n] += 1

                sample={}
                sample["relation_id"]=relation_id
                sample["images_ids"]=[]
                image_index=random.sample(range(0,images_len), bag_size[bag_n])
                for ind in image_index:
                    sample["images_ids"].append(images_ids[ind])
                samples.append(sample)
        
        return samples

    def get_relation_id_to_images(self):
        for xml_file_path in tqdm(self.all_xml_files_path, desc="reading xml files"):
            data=ET.parse(xml_file_path)
            root = data.getroot()
            img_file=root.find('filename').text
            for rel in root.findall('./relation'):
                predicate=str(rel.find("predicate").text)
                if predicate in self.predicates_name_to_id:
                    predicate_id=self.predicates_name_to_id[predicate]
                    self.relation_id_to_images[predicate_id].append(img_file)


    def __getitem__(self,idx):
        return dict(self.bag_features[idx])

    def __len__(self):
        return len(self.bag_features)
