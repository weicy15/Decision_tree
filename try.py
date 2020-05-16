import scipy.io as scio
import pandas as pd
import numpy as np


class TreeNode:
    def __init__(self):
        self.own_sample = None
        self.is_leaf = True
        self.chose_feature = None
        self.left_part = None
        self.right_part = None
        self.left_node = None
        self.right_node = None
        self.label = 0

def Impurity(samples):
    labels = samples[:, 0]
    total_num = len(labels)
    label_class = np.unique(labels).tolist()
    entropy = 0
    for target in label_class:
        p = sum(labels == target)/total_num
        entropy -= p * np.log2(p)

    return entropy


def SelectFeature(samplesUnderThisNode):
    feature_num = samplesUnderThisNode.shape[1] - 1
    sample_num = samplesUnderThisNode.shape[0]
    min_impurity = float('inf')
    best_chose_feature = None
    best_left_part = None
    best_right_part = None
    # sep_point_candidates = np.unique(samplesUnderThisNode).tolist()

    for chose_feature in range(1, feature_num+1):
        chosen_feature_of_sample = samplesUnderThisNode[:, chose_feature]
        # for sep_point in sep_point_candidates:
        index_left = np.where(chosen_feature_of_sample == 0)
        index_right = np.where(chosen_feature_of_sample == 1)
        left_part = samplesUnderThisNode[index_left]
        right_part = samplesUnderThisNode[index_right]
        if type(left_part) is not np.ndarray or type(right_part) is not np.ndarray:
            continue
        impurity_left = Impurity(left_part)
        impurity_right = Impurity(right_part)
        cur_impurity = impurity_left * left_part.shape[0] / sample_num + impurity_right * right_part.shape[0] / sample_num
        if cur_impurity < min_impurity:
            min_impurity = cur_impurity
            best_chose_feature = chose_feature
            best_left_part = left_part
            best_right_part = right_part
    max_delta_impurity = Impurity(samplesUnderThisNode) - min_impurity

    return best_chose_feature, best_left_part, best_right_part, max_delta_impurity

def SplitNode(samplesUnderThisNode, thresh):
    best_chose_feature, best_left_part, best_right_part, max_delta_impurity = SelectFeature(samplesUnderThisNode)
    print(max_delta_impurity)
    if max_delta_impurity > thresh:
        return True, best_chose_feature, best_left_part, best_right_part
    else:
        return False, None, None, None

def GenerateTree(tree, samples, thresh):
    tree.own_sample = samples
    if type(samples) is np.ndarray:
        sample_labels = samples[:, 0]
        tree.label = max(set(sample_labels.tolist()), key = sample_labels.tolist().count)
        if_sep, chose_feature, left_part, right_part = SplitNode(samples, thresh)
        if if_sep:
            tree.is_leaf = False
            tree.chose_feature = chose_feature
            tree.left_part = left_part
            tree.right_part = right_part
            tree.left_node = TreeNode()
            GenerateTree(tree.left_node, tree.left_part, thresh)
            tree.right_node = TreeNode()
            GenerateTree(tree.right_node, tree.right_part, thresh)

        else:
            tree.is_leaf = True
            tree.chose_feature = None
            tree.left_part = None
            tree.right_part = None
            tree.left_node = None
            tree.right_node = None



def Prune(GeneratedTree, CorssValidationDataset):
    print('prune11')
    labels = CorssValidationDataset[:, 0]
    if GeneratedTree.is_leaf:
        return

    if GeneratedTree.left_node.is_leaf == True and GeneratedTree.right_node.is_leaf == True:
        cross_valid_chose_feature = CorssValidationDataset[:, GeneratedTree.chose_feature]
        left_index = np.where(cross_valid_chose_feature == 0)
        right_index = np.where(cross_valid_chose_feature == 1)
        left_part = CorssValidationDataset[left_index]
        right_part = CorssValidationDataset[right_index]
        left_part_label = left_part[:, 0]
        right_part_label = right_part[:, 0]

        #计算左右正确率
        split_accuracy = (np.sum(left_part_label==GeneratedTree.left_node.label) + np.sum(right_part_label==GeneratedTree.right_node.label))/CorssValidationDataset.shape[0]

        #计算不分叉正确率
        sample_label = CorssValidationDataset[:, 0]
        unsplit_accuracy = np.sum(sample_label==GeneratedTree.label) / CorssValidationDataset.shape[0]

        if unsplit_accuracy > split_accuracy:#合并
            GeneratedTree.is_leaf = True
            GeneratedTree.chose_feature = None
            GeneratedTree.left_part = None
            GeneratedTree.right_part = None
            GeneratedTree.left_node = None
            GeneratedTree.right_node = None           


    elif GeneratedTree.left_node.is_leaf == True and GeneratedTree.right_node.is_leaf == False:
        cross_valid_chose_feature = CorssValidationDataset[:, GeneratedTree.chose_feature]
        right_index = np.where(cross_valid_chose_feature == 1)
        right_part = CorssValidationDataset[right_index]
        Prune(GeneratedTree.right_node, right_part)
        if GeneratedTree.left_node.is_leaf == True and GeneratedTree.right_node.is_leaf == True:
            cross_valid_chose_feature = CorssValidationDataset[:, GeneratedTree.chose_feature]
            left_index = np.where(cross_valid_chose_feature == 0)
            right_index = np.where(cross_valid_chose_feature == 1)
            left_part = CorssValidationDataset[left_index]
            right_part = CorssValidationDataset[right_index]
            left_part_label = left_part[:, 0]
            right_part_label = right_part[:, 0]

            #计算左右正确率
            split_accuracy = (np.sum(left_part_label==GeneratedTree.left_node.label) + np.sum(right_part_label==GeneratedTree.right_node.label))/CorssValidationDataset.shape[0]

            #计算不分叉正确率
            sample_label = CorssValidationDataset[:, 0]
            unsplit_accuracy = np.sum(sample_label==GeneratedTree.label) / CorssValidationDataset.shape[0]

            if unsplit_accuracy > split_accuracy:#合并
                GeneratedTree.is_leaf = True
                GeneratedTree.chose_feature = None
                GeneratedTree.left_part = None
                GeneratedTree.right_part = None
                GeneratedTree.left_node = None
                GeneratedTree.right_node = None

    
    elif GeneratedTree.left_node.is_leaf == False and GeneratedTree.right_node.is_leaf == True:
        cross_valid_chose_feature = CorssValidationDataset[:, GeneratedTree.chose_feature]
        left_index = np.where(cross_valid_chose_feature == 0)
        left_part = CorssValidationDataset[left_index]
        Prune(GeneratedTree.left_node, left_part)
        if GeneratedTree.left_node.is_leaf == True and GeneratedTree.right_node.is_leaf == True:
            cross_valid_chose_feature = CorssValidationDataset[:, GeneratedTree.chose_feature]
            left_index = np.where(cross_valid_chose_feature == 0)
            right_index = np.where(cross_valid_chose_feature == 1)
            left_part = CorssValidationDataset[left_index]
            right_part = CorssValidationDataset[right_index]
            left_part_label = left_part[:, 0]
            right_part_label = right_part[:, 0]

            #计算左右正确率
            split_accuracy = (np.sum(left_part_label==GeneratedTree.left_node.label) + np.sum(right_part_label==GeneratedTree.right_node.label))/CorssValidationDataset.shape[0]

            #计算不分叉正确率
            sample_label = CorssValidationDataset[:, 0]
            unsplit_accuracy = np.sum(sample_label==GeneratedTree.label) / CorssValidationDataset.shape[0]

            if unsplit_accuracy > split_accuracy:#合并
                GeneratedTree.is_leaf = True
                GeneratedTree.chose_feature = None
                GeneratedTree.left_part = None
                GeneratedTree.right_part = None
                GeneratedTree.left_node = None
                GeneratedTree.right_node = None

    else:
        cross_valid_chose_feature = CorssValidationDataset[:, GeneratedTree.chose_feature]
        left_index = np.where(cross_valid_chose_feature == 0)
        right_index = np.where(cross_valid_chose_feature == 1)
        left_part = CorssValidationDataset[left_index]
        right_part = CorssValidationDataset[right_index]
        Prune(GeneratedTree.left_node, left_part)
        Prune(GeneratedTree.right_node, right_part)
        if GeneratedTree.left_node.is_leaf == True and GeneratedTree.right_node.is_leaf == True:
            cross_valid_chose_feature = CorssValidationDataset[:, GeneratedTree.chose_feature]
            left_index = np.where(cross_valid_chose_feature == 0)
            right_index = np.where(cross_valid_chose_feature == 1)
            left_part = CorssValidationDataset[left_index]
            right_part = CorssValidationDataset[right_index]
            left_part_label = left_part[:, 0]
            right_part_label = right_part[:, 0]

            #计算左右正确率
            split_accuracy = (np.sum(left_part_label==GeneratedTree.left_node.label) + np.sum(right_part_label==GeneratedTree.right_node.label))/CorssValidationDataset.shape[0]

            #计算不分叉正确率
            sample_label = CorssValidationDataset[:, 0]
            unsplit_accuracy = np.sum(sample_label==GeneratedTree.label) / CorssValidationDataset.shape[0]

            if unsplit_accuracy > split_accuracy:#合并
                GeneratedTree.is_leaf = True
                GeneratedTree.chose_feature = None
                GeneratedTree.left_part = None
                GeneratedTree.right_part = None
                GeneratedTree.left_node = None
                GeneratedTree.right_node = None

#XToBePredicted最前面已经加上了一列0向量
def Decision(GeneratedTree, XToBePredicted, reuslt):
    if GeneratedTree.is_leaf:
        result[XToBePredicted[:, 0].tolist(), 0] = GeneratedTree.label

    else:
        XToBePredicted_chose_feature = XToBePredicted[:, GeneratedTree.chose_feature]
        left_index = np.where(XToBePredicted_chose_feature == 0)
        right_index = np.where(XToBePredicted_chose_feature == 1)
        left_part = XToBePredicted[left_index]
        right_part = XToBePredicted[right_index]
        Decision(GeneratedTree.left_node, left_part, result)
        Decision(GeneratedTree.right_node, right_part, result)



Sogou_webpage = scio.loadmat('Sogou_webpage.mat')
doclabel = Sogou_webpage['doclabel']
wordMat = Sogou_webpage['wordMat']
data = open('keyword.csv')
featureName = pd.read_csv(data, header= None)
featureName = np.array(featureName)
name = featureName[:, 1]
data = np.concatenate((doclabel, wordMat), axis= 1)
# data = data[0:5600,:]
data_len = data.shape[0]
index = np.arange(data_len)
np.random.shuffle(index)
data = data[index, :]
train_samples = data[:int(3*data_len/5), :]
cross_valid_samples = data[int(3*data_len/5):int(4*len(doclabel)/5), :]
test_samples = data[int(4*data_len/5):, :]
threshold = 0.3


tree = TreeNode()

GenerateTree(tree, train_samples, threshold)

Prune(tree, cross_valid_samples)

target_label = test_samples[:, 0]
result = np.zeros((test_samples.shape[0], 1))
test_data = np.concatenate((np.expand_dims(np.arange(0,test_samples.shape[0]), 1), test_samples[:, 1:]), axis= 1)


Decision(tree, test_data, result)

our_label = result[:, 0]
test_precise = np.sum(our_label == target_label)/ target_label.shape[0]
print("accuracy:")
print(test_precise)











        