
from typing import Text
from utils.utils import load_config_file
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import os
from collections import defaultdict 
from shutil import copyfile
from VR_SimilarityNetwork.dataloader.VrRVGDatasetTest import VrRVGDatasetTest
from VR_SimilarityNetwork.model.SimilarityNetworkVREncoder import SimilarityNetworkVREncoder
from VR_SimilarityNetwork.model.SimilarityNetworkConcat import SimilarityNetworkConcat
import cv2
from VR_Encoder.model.vtranse import VTransE
from VR_Encoder.model.concat import Concat
DATA_CONFIG_PATH = "/DATA/trevant/Vaibhav/tempVRC/VR_SimilarityNetwork/configs/data_config_test.yaml"
TESTER_CONFIG_PATH = "/DATA/trevant/Vaibhav/tempVRC/VR_SimilarityNetwork/configs//test_config.yaml"
MODEL_CONFIG_PATH = "/DATA/trevant/Vaibhav/tempVRC/VR_SimilarityNetwork/configs/model_config.yaml"

data_config = load_config_file(DATA_CONFIG_PATH)
test_config = load_config_file(TESTER_CONFIG_PATH)
model_config = load_config_file(MODEL_CONFIG_PATH)

RESULT_FOLDER = test_config.RESULT_FOLDER
BATCH_SIZE = test_config.BATCH_SIZE
RELATION_NET_CHECKPOINT = test_config.RelationNET_CHECKPOINT
SIMILARITY_NET_CONCAT_CHECKPOINT = test_config.SIMILARITY_NET_CONCAT_CHECKPOINT 
BAG_SIZE = test_config.BAG_SIZE
SIMILARITY= test_config.SIMILARITY
CONCAT = test_config.CONCAT
SAVE_OUTPUT= test_config.SAVE_OUTPUT
ANCHOR_IMAGE = test_config.ANCHOR_IMAGE
SUBJECT_ANCHORED = test_config.SUBJECT_ANCHORED
top_k= test_config.top_k

def printParams():
    print("bag size =", BAG_SIZE)
    print("similarity =", SIMILARITY)
    print("concat =", CONCAT)
    print("IMAGE ANCHORED = ", ANCHOR_IMAGE)
    print("Subject anchored= ", SUBJECT_ANCHORED)

def load_dataset(vrNetwork):
    dataset= VrRVGDatasetTest(data_config, test_config, vrNetwork)
    dataset_len=(dataset.__len__())
    print("dataset_length=",dataset_len)
    train_sz=int(0*dataset_len)
    val_size=dataset_len-train_sz
    train_dataset,val_dataset= torch.utils.data.random_split(dataset, [train_sz, val_size])
    val_dataloader= DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,collate_fn=lambda x:x)
    return val_dataloader

def save_output(relation_id, n_samples, image_ids, bboxes_sub, bboxes_obj):
    path=os.path.join(RESULT_FOLDER,str(relation_id))
    try:
        os.mkdir(path)
    except:
        pass

    path=os.path.join(path,str(n_samples))
    try:
        os.mkdir(path)
    except:
        pass
    
    for i,img in enumerate(image_ids):
        try:
            im=cv2.imread(data_config.VisualGenomeImageDir1+str(img)+".jpg")
            h,w,d=im.shape
        except:
            im=cv2.imread(data_config.VisualGenomeImageDir2+str(img)+".jpg")
            h,w,d=im.shape
        
        
        [x_min,y_min,x_max,y_max]=(bboxes_sub[i]).tolist()
        
        x_min=int(x_min*w)
        x_max=int(x_max*w)
        y_min=int(y_min*h)
        y_max=int(y_max*h)
        cv2.rectangle(im,(x_min,y_min),(x_max,y_max),(0,255,0),2)
        [x_min,y_min,x_max,y_max]=(bboxes_obj[i]).tolist()
        x_min=int(x_min*w)
        x_max=int(x_max*w)
        y_min=int(y_min*h)
        y_max=int(y_max*h)
        cv2.rectangle(im,(x_min,y_min),(x_max,y_max),(0,255,255),2)

        path_new=os.path.join(path,str(img)+".jpg")
        cv2.imwrite(path_new,im)



def binary_tree(l,r,data, net):
    mid=(l+r)//2
    if(l==r):
        ranks=[]
        tupls=[]
        relation_sz=len(data["relations"][l])
        for i in range(relation_sz):
            ranks.append(0)
            tupls.append([data["relations"][l]["relations"][i]])
        return tupls,ranks
    relations_1,rank_1=binary_tree(l,mid,data, net)
    relations_2,rank_2=binary_tree(mid+1,r,data, net)
    sz_r1=len(relations_1)
    sz_r2=len(relations_2)
    relations_final=[]
    rank_final=[]
    for i in range(sz_r1):
        for j in range(sz_r2):
            val_1=rank_1[i]
            val_2=rank_2[j]
            rela=[]
            rank=float(val_1+val_2)

            for k in range(len(relations_1[i])):
                for l in range(len(relations_2[j])):
                    tup1=(relations_1[i][k])[1]
                    tup2=relations_2[j][l][1]
                    
                    tup1=torch.tensor(tup1).cuda()
                    tup2=torch.tensor(tup2).cuda()

                    if(SIMILARITY=="cosine"):
                        cos=torch.nn.CosineSimilarity(dim=0)
                        calc = cos(tup1,tup2)

                    elif(SIMILARITY=="relation_net"):
                        calc=net(tup1,tup2)
                    
                    rank+=float(calc)
            rela=relations_1[i]+relations_2[j]
            relations_final.append(rela)
            rank_final.append(rank)
    rank_final=np.array(rank_final)
    top_k_indices=rank_final.argsort()[-top_k:][::-1]
    relations_top_k=[]
    rank_top_k=[]

    for ind in top_k_indices:
        relations_top_k.append(relations_final[ind])
        rank_top_k.append(rank_final[ind])

    return relations_top_k,rank_top_k

''' WHOLE VISUALIZE CODE IS COMMENTED OUT '''
def test(val_dataloader, net):
    n_samples=0
    n_correct=0
    n_correct_frac=0.0
    m_iou = 0.0
    n_pred=0
    n_samples_class = defaultdict(float)
    image_corloc_class = defaultdict(float)
    bag_corloc_class = defaultdict(float)

    for i_batch, data in enumerate(val_dataloader):
        get_size=len(data)
        for j in range(get_size):
            try:
                image_n=len(data[j]["relations"])
                relation_id=data[j]["relation_id"]
                relations,ranks=binary_tree(0,image_n-1,data[j], net)
                chk=0
                n_samples_class[relation_id] += 1
                for ii in range(1):
                    len_tupl=len(relations[ii])
                    verify=True
                    streak=[]
                    image_ids=[]
                    bboxes_sub=[]
                    bboxes_obj=[]
                    frac=float(1/len_tupl)
                    sum_tupl=0.0
                    for jj in range(len_tupl):
                        op=relations[ii][jj][0]
                        image_id=relations[ii][jj][3]
                        sub_iou =  relations[ii][jj][4]
                        obj_iou = relations[ii][jj][5]

                        n_pred+=1
                        m_iou += ((sub_iou-m_iou)/n_pred)
                        n_pred+=1
                        m_iou += ((obj_iou-m_iou)/n_pred)

                        image_ids.append(image_id)
                        bb_sub=relations[ii][jj][2]["sub_bnd_box"]

                        bb_obj=relations[ii][jj][2]["obj_bnd_box"]
                        bboxes_sub.append(bb_sub)
                        bboxes_obj.append(bb_obj)
                        sum_tupl+=float(op*frac)

                        streak.append(op)
                        verify=verify & op

                    if(SAVE_OUTPUT ==True):
                        save_output(relation_id, n_samples, image_ids, bboxes_sub, bboxes_obj)
                    if(verify==True):   
                        chk=1
                    
                
                n_samples+=1
                n_correct_frac+=sum_tupl
                image_corloc_class[relation_id] += sum_tupl
                if(chk==1):
                    n_correct+=1
                    bag_corloc_class[relation_id] += 1
                else:
                    bag_corloc_class[relation_id] += 0
            except:
                pass

    return n_correct, n_correct_frac, n_samples


def printResult(n_correct, n_correct_frac, n_samples):
    print("n_correct=", n_correct)
    print("n correct fraction=", n_correct_frac)
    print("n_samples=", n_samples)

def main():
    printParams()
    if test_config.CONCAT ==False:
        vrNetwork_config = load_config_file(data_config.VREncoderConfig)
        vrNetwork = VTransE(index_sp=vrNetwork_config.index_sp,
                        index_cls=vrNetwork_config.index_cls,
                        num_pred=vrNetwork_config.num_pred,
                        output_size=vrNetwork_config.output_size,
                        input_size=vrNetwork_config.input_size)
    else:
        vrNetwork = Concat()

    val_dataloader = load_dataset(vrNetwork) # loading dataset

    #loading network 
    #################################################
    if( CONCAT == False):
        net = SimilarityNetworkVREncoder(model_config)
        net = net.cuda()
        chkpt=torch.load(RELATION_NET_CHECKPOINT) 

    else:
        net = SimilarityNetworkConcat(model_config)
        net = net.cuda()
        chkpt=torch.load(SIMILARITY_NET_CONCAT_CHECKPOINT)

    net.load_state_dict(chkpt["model"])
    net.eval() 
    torch.no_grad()
    ##################################################

    n_correct, n_correct_frac, n_samples =test(val_dataloader, net)
    printResult(n_correct, n_correct_frac, n_samples)
    
if __name__ == "__main__":
    main()



