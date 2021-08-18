import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utils import load_config_file
from VR_SimilarityNetwork.model.SimilarityNetworkConcat import SimilarityNetworkConcat 
from VR_SimilarityNetwork.model.SimilarityNetworkVREncoder import SimilarityNetworkVREncoder 
from VR_SimilarityNetwork.dataloader.VrRVGDatasetTrain import VrRVGDatasetTrain
from VR_Encoder.model.vtranse import VTransE
from VR_Encoder.model.concat import Concat
from tqdm import tqdm
import time

DATA_CONFIG_PATH = "/DATA/trevant/Vaibhav/tempVRC/VR_SimilarityNetwork/configs/data_config_train.yaml"
TRAINER_CONFIG_PATH = "/DATA/trevant/Vaibhav/tempVRC/VR_SimilarityNetwork/configs//train_config.yaml"
MODEL_CONFIG_PATH = "/DATA/trevant/Vaibhav/tempVRC/VR_SimilarityNetwork/configs/model_config.yaml"

#######################################
# Defining the loss
def episodic_loss(r, R):
    return torch.log(1+torch.exp(-R*r))
#######################################

def save_checkpoint(checkpoint, train_config):
    time.sleep(10)
    path = train_config.NETWORK + '_checkpoint.pth'
    torch.save(checkpoint, path)

def per_sample_training(bag_size, ith_bag, net):
    cnt_pairs = 0
    bag_loss = 0.0

    for j in range(bag_size):
        image_ind_1=j
        image_ind_2=(j+1)%bag_size

        n_positive_1=len(ith_bag["relations"][image_ind_1]["positive_relations"]) # count positive relations in image 1
        n_negative_1=len(ith_bag["relations"][image_ind_1]["negative_relations"]) # count negative relations in image 1
        n_positive_2=len(ith_bag["relations"][image_ind_2]["positive_relations"]) # count positive relations in image 2
        n_negative_2=len(ith_bag["relations"][image_ind_2]["negative_relations"])# count negative relations in image 2

        cnt=0
        for a in range(n_positive_1):
            if(cnt > 10):
                break
            for b in range(n_positive_2):
                cnt+=1
                positive_example_1=torch.tensor(ith_bag["relations"][image_ind_1]["positive_relations"][a]).cuda()
                positive_example_2=torch.tensor(ith_bag["relations"][image_ind_2]["positive_relations"][b]).cuda()
                label=1
                r=net(positive_example_1,positive_example_2)

                loss=episodic_loss(r,label)
                cnt_pairs+=1
                bag_loss+=loss

        sample=cnt//2
        itr=0
        for a in range(n_positive_1):
            for b in range(n_negative_2):
                if itr>sample:
                    break
                itr+=1
                positive_example_1=torch.tensor(ith_bag["relations"][image_ind_1]["positive_relations"][a]).cuda()
                negative_example_2=torch.tensor(ith_bag["relations"][image_ind_2]["negative_relations"][b]).cuda()
                label=-1
                r=net(positive_example_1,negative_example_2)

                loss=episodic_loss(r,label)
                bag_loss+=loss
                cnt_pairs+=1
        
        itr=0
        for a in range(n_positive_2):
            for b in range(n_negative_1):
                if itr>sample:
                    break
                itr+=1
                positive_example_1=torch.tensor(ith_bag["relations"][image_ind_2]["positive_relations"][a]).cuda()
                negative_example_2=torch.tensor(ith_bag["relations"][image_ind_1]["negative_relations"][b]).cuda()
                label=-1
                r=net(positive_example_1,negative_example_2)
                
                loss=episodic_loss(r,label)
                bag_loss+=loss
                cnt_pairs+=1
        
    return bag_loss, cnt_pairs

def per_epoch_train(net, train_dataloader, optimizer, scheduler):
    epoch_loss = 0.0
    
    for batch_data in tqdm(train_dataloader, desc="Training an epoch"):
        batch_size=len(batch_data)
        for i in range(batch_size):
            ith_bag=batch_data[i]
            bag_size=len(ith_bag["relations"])
            optimizer.zero_grad()
            bag_loss, cnt_pairs = per_sample_training(bag_size , ith_bag, net)
            
            bag_loss/=cnt_pairs
            bag_loss.backward()
            optimizer.step()

            epoch_loss+=bag_loss.item()
        
    scheduler.step(epoch_loss)
    
    return epoch_loss

def train(train_config, dataset, net):
    epochs = train_config.epochs
    batch_size=train_config.batch_size

    optimizer = optim.Adam(net.parameters())

    train_dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0, collate_fn=lambda x:x)

    scheduler = ReduceLROnPlateau(optimizer, mode= train_config.scheduler.mode , factor=train_config.scheduler.factor, patience= train_config.scheduler.patience, verbose= train_config.scheduler.verbose)
    net.train()

    epoch_loss_min=100000000
    for epoch in range(epochs):
        epoch_loss = per_epoch_train(net, train_dataloader, optimizer, scheduler)
        print('Epoch  input: {} \tTraining Loss: {:.6f} '.format(epoch, epoch_loss))
    
        if epoch_loss < epoch_loss_min:
            print('training loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(epoch_loss_min,epoch_loss))
            checkpoint = { 
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict()}

            print("dataset sample size=",len(dataset))
            save_checkpoint(checkpoint,train_config)
            epoch_loss_min=epoch_loss

def main():

    data_config = load_config_file(DATA_CONFIG_PATH)
    train_config = load_config_file(TRAINER_CONFIG_PATH)
    model_config = load_config_file(MODEL_CONFIG_PATH)


    if data_config.VREncoderEmbeddings == 'VTransE':
        vrNetwork_config = load_config_file(data_config.VREncoderConfig)
        vrNetwork = VTransE(index_sp=vrNetwork_config.index_sp,
                        index_cls=vrNetwork_config.index_cls,
                        num_pred=vrNetwork_config.num_pred,
                        output_size=vrNetwork_config.output_size,
                        input_size=vrNetwork_config.input_size)
    elif data_config.VREncoderEmbeddings == 'VRConcat':
        vrNetwork = Concat()

    dataset = VrRVGDatasetTrain(data_config, vrNetwork)
    dataset_len=(dataset.__len__())
    print("dataset_length=",dataset_len)
    
    if( train_config.NETWORK == "SimilarityNetworkVREncoder"):
        net = SimilarityNetworkVREncoder(model_config)

    if( train_config.NETWORK == "SimilarityNetworkConcat"):
        net = SimilarityNetworkConcat(model_config)

    net = net.cuda()
    
    train(train_config, dataset, net)
    print("Training Done")

if __name__ == "__main__":
    main()
