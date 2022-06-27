import torch
from torch.distributions import Categorical


@torch.no_grad()
def compute_label_transform_matrix(labels_t1, labels_t2):
    assert labels_t1.size(1) == labels_t2.size(1) # make sure sample num are equal
    sample_num = labels_t1.size(1)
    class_num_t1 = labels_t1.unique().size(0)
    class_num_t2 = labels_t2.unique().size(0)
    dual_labels = torch.cat((labels_t1, labels_t2),0).t()
    label_tran_mat = torch.zeros(class_num_t1, class_num_t2)
    for x in dual_labels:
        label_tran_mat[x[0].item(), x[1].item()] += 1
    return label_tran_mat


@torch.no_grad()
def compute_inclass_distributiions(labels_t1, labels_t2, dis_type="original"):
    label_tran_mat = compute_label_transform_matrix(labels_t1, labels_t2)
    if dis_type=="softmax":
    	return torch.nn.functional.softmax(label_tran_mat, 0)
    else:
        return label_tran_mat / label_tran_mat.sum(0)


# Method 1 as class weights
@torch.no_grad()
def compute_class_stablization_by_entropy(labels_t1, labels_t2):
    distributions = compute_inclass_distributiions(labels_t1, labels_t2, dis_type="softmax")
    return Categorical(probs = distributions.t()).entropy()




@torch.no_grad()
def compute_label_iou_matrix(labels_t1, labels_t2):
    class_num_t1 = labels_t1.unique().size(0)
    class_num_t2 = labels_t2.unique().size(0)
    dual_labels = torch.cat((labels_t1, labels_t2),0).t()
    label_union_mat_1 = torch.zeros(class_num_t1, class_num_t2)
    label_union_mat_2 = torch.zeros(class_num_t1, class_num_t2).t()
    for x in dual_labels:
        label_union_mat_1[x[0].item()] += 1
        label_union_mat_2[x[1].item()] += 1
    label_inter_mat = compute_label_transform_matrix(labels_t1, labels_t2)
    label_union_mat = label_union_mat_1 + label_union_mat_2.t() - label_inter_mat
    return label_inter_mat / label_union_mat




@torch.no_grad()
def compute_sample_weights(labels_t1, labels_t2):
    ioumat = torch.nn.functional.softmax(compute_label_iou_matrix(labels_t1, labels_t2), 1)
    return torch.index_select(torch.index_select(ioumat, 0, labels_t1[0])[0], 0, labels_t2[0])


@torch.no_grad()
def compute_class_softlabels(labels_t1, labels_t2, matr_type="trans", distr_type="original"):
    assert labels_t1.size(1) == labels_t2.size(1) # make sure sample num are equal
    if matr_type == "trans":
        matr = compute_label_transform_matrix(labels_t1, labels_t2)
    else:
        matr = compute_label_iou_matrix(labels_t1, labels_t2)
    if distr_type=="original":
        return (matr.t() / matr.t().sum(0)).t()
    else:
        return torch.nn.functional.softmax(matr, 1)



@torch.no_grad()
def compute_sample_softlabels(labels_t1, labels_t2, matr_type="trans", distr_type="original"):
    class_softlabels = compute_class_softlabels(labels_t1, labels_t2, matr_type, distr_type)
    return torch.index_select(class_softlabels, 0, labels_t1[0])