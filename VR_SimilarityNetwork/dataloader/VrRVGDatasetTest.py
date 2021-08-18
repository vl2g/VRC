import os
import numpy as np
from torch.utils.data import Dataset
import glob
import xml.etree.ElementTree as ET
from collections import defaultdict
import glob
import random
from tqdm import tqdm
from scipy import special
from utils.utils import read_json
from utils.sampling_utils import get_iou

import torch

class VrRVGDatasetTest(Dataset):
    def __init__(self, data_config, test_config, vrNetwork):
        self.data_config = data_config
        self.test_config = test_config
        
        self.vrNetwork = vrNetwork
        self.vrNetwork = self.vrNetwork.cuda()
        if(self.test_config.CONCAT== False):
            chkpt=torch.load(self.data_config.VREncoder_Net_Checkpoint) 
            self.vrNetwork.load_state_dict(chkpt["model_state_dict"])
        self.vrNetwork.eval()

        self.xml_path = self.data_config.XML_FILE_PATH_VrRVG
        self.all_xml_files_path=glob.glob(self.xml_path+"/*.xml")
        self.npy_file_path = self.data_config.NPY_FILE_PATH
        
        self.bag_size = self.test_config.BAG_SIZE
        self.predicates_name_to_id=read_json(data_config.test_predicates_path)
        self.relation_id_to_images=defaultdict(list)
        self.get_relation_id_to_images()
        self.sampl=self.load_samples()
        self.samples=self.test_sample_generator()

    def extract_vtranse_embedding_from_relation(self,info,feat,subjs,objs,image_width,image_height, subj_bbox, obj_bbox):
        relation={}
        bnd_boxx=info.item().get('bbox')[subjs] # [xmin ymin xmax ymax]
        bnd_box=bnd_boxx.copy()
        sub_pred = {}
        sub_pred["xmin"] = bnd_box[0]
        sub_pred["ymin"] = bnd_box[1]
        sub_pred["xmax"] = bnd_box[2]
        sub_pred["ymax"] = bnd_box[3]
        sub_iou = get_iou(sub_pred,subj_bbox)

        bnd_box[0]=float(bnd_box[0]/image_width)
        bnd_box[2]/=image_width
        bnd_box[1]/=image_height
        bnd_box[3]/=image_height
        relation["sub_class_scores"] = info.item()["class_scores"][subjs]
        relation["obj_class_scores"]= info.item()["class_scores"][objs]

        relation["sub_bnd_box"]=bnd_box
        
        bnd_boxxx=info.item().get('bbox')[objs] # [xmin ymin xmax ymax]
        bnd_box=bnd_boxxx.copy()
        obj_pred = {}
        obj_pred["xmin"] = bnd_box[0]
        obj_pred["ymin"] = bnd_box[1]
        obj_pred["xmax"] = bnd_box[2]
        obj_pred["ymax"] = bnd_box[3]
        obj_iou = get_iou(obj_pred,obj_bbox)
        
        bnd_box[0]/=image_width
        bnd_box[2]/=image_width
        bnd_box[1]/=image_height
        bnd_box[3]/=image_height
        relation["obj_bnd_box"]=bnd_box
        
        relation["sub_roi_features"]=feat[subjs]
        relation["obj_roi_features"]=feat[objs]
        if(self.test_config.CONCAT == 0):
            embedding=self.get_vtranse_embedding(relation)
        else:
            embedding=self.get_concat_embedding(relation)

        return embedding,relation, sub_iou, obj_iou


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
        

    def get_concat_embedding(self, sample_batched):
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
        
    def test_sample_generator(self):
        samples=[]
        positive_examples=0
        negative_examples=0
        cnt=0
        for sample in tqdm(self.sampl, desc =" iterating samples"):
            sample_relation_id=sample["relation_id"]
            relations_per_sample={}
            relations_per_sample["relation_id"]=sample_relation_id
            relations_per_sample["relations"]=[] # n1,n2,n3 relations in images
            for img_ind,img in enumerate(sample["images_ids"]):
                relations_per_image={}
                img_name=img.split(".")[0]

                relations_per_image["image_name"]=img_name
                relations_per_image["relations"]=[]

                npy_info_name=img_name+"_info.npy"
                npy_feat_name=img_name+".npy"
                info=np.load(os.path.join(self.npy_file_path,npy_info_name),allow_pickle=True)
                feat=np.load(os.path.join(self.npy_file_path,npy_feat_name),allow_pickle=True)
                xml_file_path=img_name+".xml"
                xml_file_path = os.path.join(self.xml_path, xml_file_path)

                data=ET.parse(xml_file_path)
                root = data.getroot()

                positive_relations_id_from_xml=[]
                all_relations_in_xml = root.findall('./relation')
                
                # function of this for loop : finding positive relations in xml
                for rel in all_relations_in_xml:
                    predicate=str(rel.find("predicate").text)
                    verify_test_train = self.predicates_name_to_id.get(predicate,None)
                    if verify_test_train is None :
                        continue
                    
                    if(verify_test_train == sample_relation_id):
                        rel_sub_id=str(rel.find('./subject_id').text)
                        rel_obj_id=str(rel.find('./object_id').text)
                        relation_key=rel_sub_id+rel_obj_id
                        positive_relations_id_from_xml.append(relation_key)
                    
                    
                for rel in all_relations_in_xml:
                    predicate=str(rel.find("predicate").text)
                    check=self.predicates_name_to_id.get(predicate, None)
                    if check is None :
                        continue
                    if(check==sample_relation_id):
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

                                    # get subject and object bounding box for xml files 
                                    subject_bbox["xmin"]=float(sub.find('bndbox').find('xmin').text)
                                    subject_bbox["xmax"]=float(sub.find('bndbox').find('xmax').text)
                                    subject_bbox["ymin"]=float(sub.find('bndbox').find('ymin').text)
                                    subject_bbox["ymax"]=float(sub.find('bndbox').find('ymax').text)

                                    object_bbox["xmin"]=float(obj.find('bndbox').find('xmin').text)
                                    object_bbox["xmax"]=float(obj.find('bndbox').find('xmax').text)
                                    object_bbox["ymin"]=float(obj.find('bndbox').find('ymin').text)
                                    object_bbox["ymax"]=float(obj.find('bndbox').find('ymax').text)

                                    #get subjects and objects roi index from npy file where the iou is greater than the threshold for subjects and objects

                                    subject_roi_index, neutral_sub_roi_index = self.get_subject_roi_index(root, subject_bbox, info, sample_relation_id, positive_relations_id_from_xml, all_relations_in_xml, type="subject")
                                    object_roi_index, neutral_obj_roi_index = self.get_subject_roi_index(root, object_bbox, info, sample_relation_id, positive_relations_id_from_xml, all_relations_in_xml, type="object")

                                    image_width = int(info.item().get('image_width'))
                                    image_height = int(info.item().get('image_height'))
                                    sampling=0

                                    
                                    if(self.test_config.ANCHOR_IMAGE == 1 and img_ind==0):
                                        for i,subjs in enumerate(subject_roi_index):
                                            for j,objs in enumerate(object_roi_index):
                                                embedding,relation, iou_sub, iou_obj = self.extract_vtranse_embedding_from_relation(info,feat,subjs,objs,image_width,image_height, subject_bbox, object_bbox)
                                                relations_per_image["relations"].append((1,embedding,relation,img_name, iou_sub, iou_obj)) 
                                                positive_examples+=1
                                                sampling+=1
                                                
                                    else:
                                        for i,subjs in enumerate(subject_roi_index):
                                            for j,objs in enumerate(object_roi_index):
                                                embedding,relation, iou_sub, iou_obj = self.extract_vtranse_embedding_from_relation(info,feat,subjs,objs,image_width,image_height, subject_bbox, object_bbox)
                                                relations_per_image["relations"].append((1,embedding,relation,img_name, iou_sub, iou_obj))  
                                                positive_examples+=1
                                                sampling+=1
                                        rois_info=info.item().get('bbox')
                                        rois=rois_info.shape[0]
                                        objs_len=len(object_roi_index)
                                        subjs_len=len(subject_roi_index)
                                        # for negative pair : positive sub - negative-obj
                                        for i,subjs in enumerate(subject_roi_index):
                                            for objs in range(0,rois):
                                                if(objs not in object_roi_index):
                                                    embedding,relation, iou_sub, iou_obj = self.extract_vtranse_embedding_from_relation(info,feat,subjs,objs,image_width,image_height, subject_bbox, object_bbox)
                                                    relations_per_image["relations"].append((0,embedding,relation,img_name, iou_sub, iou_obj)) 
                                                    negative_examples+=1
                                                    sampling+=1
                                        
                                        if(self.test_config.SUBJECT_ANCHORED==False):
                                            # for negative pair : negative sub - positive-obj
                                            for subjs in range(0,rois):
                                                for j,objs in enumerate(object_roi_index):
                                                    if(subjs not in subject_roi_index ):
                                                        embedding,relation, iou_sub, iou_obj = self.extract_vtranse_embedding_from_relation(info,feat,subjs,objs,image_width,image_height, subject_bbox, object_bbox)
                                                        relations_per_image["relations"].append((0,embedding,relation,img_name, iou_sub, iou_obj)) 
                                                        negative_examples+=1
                                                        sampling+=1
                                            
                                            # taking negative pairs : negative sub - negative obj 
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
                                                            embedding,relation, iou_sub, iou_obj = self.extract_vtranse_embedding_from_relation(info,feat,subjs,objs,image_width,image_height, subject_bbox, object_bbox)
                                                            # print("appending negative-negative pairs relations in img")
                                                            negative_examples+=1
                                                            relations_per_image["relations"].append((0,embedding,relation,img_name, iou_sub, iou_obj))  
                                                            sampling+=1
                                        
                
                if(len(relations_per_image["relations"])!=0 ):
                    # print("adding relations of 1 image in bag")
                    #print('len(relations_per_image["relations"])', len(relations_per_image["relations"]))

                    relations_per_sample["relations"].append(relations_per_image)
                # else :
                #     #print('len(relations_per_image["relations"])', len(relations_per_image["relations"]))
        
            if(len(relations_per_sample["relations"])==self.bag_size):
                # print()
                cnt += 1
                
                samples.append(relations_per_sample)  

        print("positive_examples=",positive_examples)
        print("negative_examples=",negative_examples)
        return samples

    def get_subject_roi_index(self, root, bbox, info, bag_relation_id, positive_relations_id_from_xml, all_relations_in_xml, type="subject"):
        
        indexes=[]
        neutral_indexes=set()

        rois_info=info.item().get('bbox')
        rois=rois_info.shape[0]
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
                check= self.predicates_name_to_id.get(predicate, None)

                if check is None:
                    continue
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
        for relation_id,images_ids in tqdm(self.relation_id_to_images.items(), desc="returing a list of dictionary og relations"):
            images_len=len(images_ids)
            assert images_len>=self.bag_size
            sample_per_relation=int((special.comb(images_len,2)))
            
            if(sample_per_relation > self.data_config.SAMPLE_PER_RELATION):
                sample_per_relation= self.data_config.SAMPLE_PER_RELATION
            for j in range(0,sample_per_relation):
                sample={}
                sample["relation_id"]=relation_id
                sample["images_ids"]=[]
                image_index=random.sample(range(0,images_len),self.bag_size)
                for ind in image_index:
                    sample["images_ids"].append(images_ids[ind])
                samples.append(sample)
        return samples

    def get_relation_id_to_images(self):
        for xml_file_path in tqdm(self.all_xml_files_path, desc ="reading xml"):
            data=ET.parse(xml_file_path)
            root = data.getroot()
            img_file=root.find('filename').text
            for rel in root.findall('./relation'):
                predicate=str(rel.find("predicate").text)
                
                if predicate in self.predicates_name_to_id:
                    predicate_id=self.predicates_name_to_id[predicate]
                    self.relation_id_to_images[predicate_id].append(img_file)

    def __getitem__(self,idx):
        return dict(self.samples[idx])

    def __len__(self):
        return len(self.samples)
