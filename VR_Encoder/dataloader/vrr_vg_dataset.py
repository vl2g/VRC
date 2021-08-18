import os
import numpy as np
from torch.utils.data import Dataset
import glob
import xml.etree.ElementTree as ET
from utils.utils import read_json, mkdir
from utils.sampling_utils import get_roi_index
from tqdm import tqdm

# TODO:
# 1. remove cnt variables in the end after complete debugging
# 2. role of self.class_imbalance


class VrRVG_train_dataset(Dataset):
    def __init__(self, xml_file_path, npy_file_path, saved_vtranse_input, saved_dir, train_predicates_path):
        self.xml_path = xml_file_path
        self.npy_file_path = npy_file_path
        self.relations = []
        self.predicates_name_to_id = read_json(train_predicates_path)
        self.class_imbalance = np.zeros((101))

        self.saved_dir = saved_dir
        mkdir(self.saved_dir)

        if saved_vtranse_input:
            self.loadrelations()
        else:
            self.getrelations()
        self.print_class_imbalance()

    def print_class_imbalance(self):
        np.save("class_imbalance.npy", self.class_imbalance)
        for i, val in enumerate(self.class_imbalance):
            print("class #", i, " and freq=", val)
        return

    def loadrelations(self):
        lis = os.listdir(self.saved_dir)
        for item in tqdm(lis, desc="loading relations..."):
            try:
                item_path = os.path.join(self.saved_dir, item)
                arr = np.load(item_path, allow_pickle=True)
                arr = arr.item()
                predic_id = int(arr["predicate"])

                if(self.class_imbalance[predic_id] < 500):
                    self.class_imbalance[predic_id] += 1
                    self.relations.append(arr)
            except:
                pass
        return

    def getrelations(self):
        cnt_relations = 0
        all_xml_files_path = glob.glob(self.xml_path+"/*.xml")
        for i, xml_file_path in enumerate(all_xml_files_path):
            if i % 1000 == 999:
                print("file #", i)
                print("dataset size=", len(self.relations))
            data = ET.parse(xml_file_path)
            root = data.getroot()
            img_file = root.find('filename').text
            img_name = img_file.split(".")[0]
            npy_info_name = img_name+"_info.npy"
            npy_feat_name = img_name+".npy"
            try:
                info = np.load(os.path.join(self.npy_file_path,
                               npy_info_name), allow_pickle=True)
                feat = np.load(os.path.join(self.npy_file_path,
                               npy_feat_name), allow_pickle=True)
            except:
                continue

            for sub in root.findall('./object'):
                for obj in root.findall('./object'):
                    sub_id = int(sub.find('object_id').text)
                    obj_id = int(obj.find('object_id').text)
                    relation = {}
                    for rel in root.findall('./relation'):
                        predicate = str(rel.find("predicate").text)
                        try:
                            rel_sub_id = int(rel.find('./subject_id').text)
                            rel_obj_id = int(rel.find('./object_id').text)

                            if(rel_sub_id == sub_id and rel_obj_id == obj_id):
                                subject_bbox = {}
                                object_bbox = {}

                                subject_bbox["xmin"] = float(
                                    sub.find('bndbox').find('xmin').text)
                                subject_bbox["xmax"] = float(
                                    sub.find('bndbox').find('xmax').text)
                                subject_bbox["ymin"] = float(
                                    sub.find('bndbox').find('ymin').text)
                                subject_bbox["ymax"] = float(
                                    sub.find('bndbox').find('ymax').text)

                                object_bbox["xmin"] = float(
                                    obj.find('bndbox').find('xmin').text)
                                object_bbox["xmax"] = float(
                                    obj.find('bndbox').find('xmax').text)
                                object_bbox["ymin"] = float(
                                    obj.find('bndbox').find('ymin').text)
                                object_bbox["ymax"] = float(
                                    obj.find('bndbox').find('ymax').text)

                                predicate_id = self.predicates_name_to_id[predicate]
                                relation["predicate"] = predicate_id

                                subject_roi_index, subj_roi_iou = get_roi_index(
                                    subject_bbox, info)
                                object_roi_index, obj_roi_iou = get_roi_index(
                                    object_bbox, info)

                                image_width = int(
                                    info.item().get('image_width'))
                                image_height = int(
                                    info.item().get('image_height'))
                                for i, subjs in enumerate(subject_roi_index):
                                    for j, objs in enumerate(object_roi_index):
                                        results = {}
                                        results.update(relation)
                                        bnd_boxx = info.item().get(
                                            'bbox')[subjs]  # [xmin ymin xmax ymax]
                                        bnd_box = bnd_boxx.copy()

                                        bnd_box[0] = float(
                                            bnd_box[0]/image_width)
                                        bnd_box[2] /= image_width
                                        bnd_box[1] /= image_height
                                        bnd_box[3] /= image_height

                                        results["sub_bnd_box"] = bnd_box
                                        results["sub_roi_iou"] = subj_roi_iou[i]
                                        results["obj_roi_iou"] = obj_roi_iou[j]
                                        results["sub_class_scores"] = info.item(
                                        )["class_scores"][subjs]  # 1601-d vector
                                        results["obj_class_scores"] = info.item(
                                        )["class_scores"][objs]  # 1601-d vector

                                        bnd_boxxx = info.item().get(
                                            'bbox')[objs]  # [xmin ymin xmax ymax]
                                        bnd_box = bnd_boxxx.copy()
                                        bnd_box[0] /= image_width
                                        bnd_box[2] /= image_width
                                        bnd_box[1] /= image_height
                                        bnd_box[3] /= image_height
                                        results["obj_bnd_box"] = bnd_box

                                        results["sub_roi_features"] = feat[subjs]
                                        results["obj_roi_features"] = feat[objs]

                                        self.relations.append(results)
                                        filename = os.path.join(
                                            self.saved_dir, str(cnt_relations)+".npy")
                                        with open(filename, 'wb') as f:
                                            np.save(
                                                f, results, allow_pickle=True)
                                            cnt_relations += 1

                        except:
                            pass

    def __len__(self):
        return len(self.relations)

    def __getitem__(self, idx):
        return self.relations[idx]
