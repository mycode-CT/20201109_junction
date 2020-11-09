import numpy as np
import sys # chin

class Ellipsoid():
    def __init__(self, x_com=0, y_com=0, z_com=0, fa=0, lambda_maj=1, x_vec=0, y_vec=0):
    #def __init__(self, x_com, y_com, z_com, fa, lambda_maj, x_vec, y_vec):
        self.x_com = x_com
        self.y_com = y_com
        self.z_com = z_com
        self.fa = fa
        self.lambda_maj = lambda_maj
        self.x_vec = x_vec
        self.y_vec = y_vec

    def find_lambdaratio_fromfa(self):
        '''
        Determines the ratio between lambda2 = lambda3 and lambda1 from FA value
        :return:
        '''
        b = -2
        fa2 = self.fa**2
        delta = 4 * fa2 * (3 - 2*fa2)
        a = 1 - 2*fa2

        sol1 = (-1 * b + np.sqrt(delta)) / (2*a)
        sol2 = (-1 * b - np.sqrt(delta)) / (2*a)
        #print(sol1, sol2)
        sol = np.where(sol1 < 1, np.maximum(sol1, sol2), sol2)
        sol = np.where(sol < 0, np.zeros_like(sol), sol)
        return sol

    def find_angles_fromvect(self):
        '''
        Determines theta and phi angles from unit vector determined by x_vec
        and y_vec
        :return:
        '''
        x = self.x_vec
        y = self.y_vec
        theta = np.arctan(y/np.maximum(x, 1e-20))
        z = np.sqrt(np.maximum(1-x**2 - y**2, 0))
        phi = np.arctan(np.sqrt(x**2+y**2)/np.maximum(z, 1e-20))
        #print(theta, phi)
        return theta, phi

    def create_mask_fromell(self, shape):
        '''
        Create map of segmented ellipse from ellipsoid characteristics and
        image shape
        :param shape:
        :return:
        '''
        theta, phi = self.find_angles_fromvect()
        x_lin = np.arange(0,shape[0])
        y_lin = np.arange(0,shape[1])
        z_lin = np.arange(0,shape[2])
        x, y, z = np.meshgrid(x_lin, y_lin, z_lin)
        s = (x - self.x_com) * np.cos(phi)*np.cos(theta) + (y - self.y_com) * np.sin(theta)*np.cos(phi) + (z - self.z_com) * np.sin(phi)
        t = (y - self.y_com)*np.cos(theta) - (x - self.x_com) * np.sin(theta)
        u = (z - self.z_com)*np.cos(phi) - (y - self.y_com) * np.sin(theta)*np.sin(phi) - (x - self.x_com) * np.cos(theta) * np.sin(phi)
        a = self.lambda_maj
        ratio = self.find_lambdaratio_fromfa()
        b = ratio*a
        c = ratio*a
        ellipsoid_app = np.square(s)/(a*a) + np.square(t)/(b*b) + np.square(u)/(c*c)
        ellipse_in = np.where(ellipsoid_app <= 1, np.ones(shape), np.zeros(shape))
        #print(np.sum(ellipse_in), self.lambda_maj, self.x_com, self.y_com, self.z_com)
        return ellipse_in

    def create_box_fromell(self, shape):
        mask = self.create_mask_fromell(shape)
        #print(np.sum(mask))
        if np.sum(mask) == 0:
            return 0, None
        else:
            indices = np.asarray(np.where(mask>0)).T
            min_ind = np.min(indices,0)
            max_ind = np.max(indices,0)
            return np.sum(mask), np.concatenate([min_ind, max_ind])

    def correction_ellipse(self, correction):
        '''
        Correction of an ellipse given correction (or how to apply correction after RPN or RCNN)
        :param correction:
        :return:
        '''
        self.x_com += correction[0]
        self.y_com += correction[1]
        self.z_com += correction[2]
        self.lambda_maj *= correction[3]
        self.fa *= correction[4]
        self.x_vec += correction[5]
        self.y_vec += correction[6]


    @staticmethod
    def hellinger_overlap_2par(ellipse1, ellipse2):
        mu_1 = np.asarray([ellipse1.x_com, ellipse1.y_com, ellipse1.z_com])
        mu_2 = np.asarray([ellipse2.x_com, ellipse2.y_com, ellipse2.z_com])

        lratio_1 = ellipse1.find_lambdaratio_fromfa()
        lratio_2 = ellipse2.find_lambdaratio_fromfa()

        rotation_1 = ellipse1.create_rotation_mat()
        rotation_2 = ellipse2.create_rotation_mat()

        diag_1 = np.diag([ellipse1.lambda_maj, lratio_1 * ellipse1.lambda_maj, lratio_1 * ellipse1.lambda_maj])
        diag_2 = np.diag([ellipse2.lambda_maj, lratio_2 * ellipse2.lambda_maj, lratio_2 * ellipse2.lambda_maj])

        cov1 = np.matmul(rotation_1, np.matmul(diag_1, rotation_1.T))
        cov2 = np.matmul(rotation_2, np.matmul(diag_2, rotation_2.T))

        h_dist_mult = np.power(np.linalg.det(cov1) * np.linalg.det(cov2), 0.25)/ np.sqrt(np.linalg.det((cov1+cov2)/2))
        h_dist_exp = -1.0/8 * np.matmul(mu_1 - mu_2, np.matmul(np.linalg.inv((cov1+cov2)/2), np.asarray(mu_1-mu_2).T))
        return 1 - h_dist_mult * np.exp(h_dist_exp)

    @staticmethod
    def hellinger_pairwise(list_ellipses):

        len_1 = len(list_ellipses)
        print(len_1)
        mu_1 = np.zeros([len(list_ellipses), 3])
        lratio_1 = np.zeros(len(list_ellipses))
        diag_prep1 = np.zeros((len_1, 9))
        rotation_1 = np.zeros([len_1, 3, 3])

        for (i, ellipse1) in enumerate(list_ellipses):
            mu_1[i, :] = np.asarray([ellipse1.x_com, ellipse1.y_com, ellipse1.z_com])
            lratio_1[i] = ellipse1.find_lambdaratio_fromfa()
            if ellipse1.lambda_maj < 0:
                ellipse1.lambda_maj = ellipse1.lambda_maj*-1
            diag_prep1[i, 0] = ellipse1.lambda_maj
            diag_prep1[i, 4] = lratio_1[i] * ellipse1.lambda_maj
            diag_prep1[i, 8] = lratio_1[i] * ellipse1.lambda_maj

        diag_1 = np.reshape(diag_prep1, [-1, 3, 3])
        rotation_1 = create_rotation_mat(list_ellipses)
        cov1 = np.matmul(rotation_1, np.matmul(diag_1, np.transpose(rotation_1,(0, 2, 1))))
        h_dist_mult = np.power(np.linalg.det(cov1)[:, None, ...] * np.linalg.det(cov1), 0.25) / np.sqrt(np.linalg.det((cov1[:, None, ...] + cov1) / 2))
        h_dist_exp = -1.0 / 8 * np.matmul((mu_1[:, None, ...] - mu_1)[..., None, :], np.matmul(np.linalg.inv((cov1[:, None, ...] + cov1) / 2), np.transpose((mu_1[:, None, :] - mu_1)[..., None, :], (0, 1, 3, 2))))

        hh = 1 - h_dist_mult * np.exp(np.squeeze(h_dist_exp))
        return hh




    @staticmethod
    def hellinger_overlap(list_ellipses1, list_ellipses2):

        """
        Hellinger distance between two ellipsoid defined by list of
        parametrisation par1 and par2. x y z lambda1 fa xvec yvec
        :param par1:
        :param par2:
        :return:
        """
        len_1 = len(list_ellipses1)
        len_2 = len(list_ellipses2)

        mu_1 = np.zeros([len(list_ellipses1),3])
        mu_2 = np.zeros([len(list_ellipses2), 3])
        lratio_1 = np.zeros(len(list_ellipses1))
        lratio_2 = np.zeros(len(list_ellipses2))
        diag_prep1 = np.zeros((len_1, 9))
        diag_prep2 = np.zeros([len_2, 9])
        rotation_1 = np.zeros([len_1, 3, 3])
        rotation_2 = np.zeros([len_2, 3, 3])
        for (i, ellipse1) in enumerate(list_ellipses1):
            mu_1[i, :] = np.asarray([ellipse1.x_com, ellipse1.y_com, ellipse1.z_com])
            lratio_1[i] = ellipse1.find_lambdaratio_fromfa()
            diag_prep1[i, 0] = ellipse1.lambda_maj
            diag_prep1[i, 4] = lratio_1[i] * ellipse1.lambda_maj
            diag_prep1[i, 8] = lratio_1[i] * ellipse1.lambda_maj

        for (i, ellipse2) in enumerate(list_ellipses2):
            mu_2[i, :] = np.asarray([ellipse2.x_com, ellipse2.y_com, ellipse2.z_com])
            lratio_2[i] = ellipse2.find_lambdaratio_fromfa()
            diag_prep2[i, 0] = ellipse2.lambda_maj
            diag_prep2[i, 4] = lratio_2[i] * ellipse2.lambda_maj
            diag_prep2[i, 8] = lratio_2[i] * ellipse2.lambda_maj

        diag_1 = np.reshape(diag_prep1,[-1, 3, 3])
        diag_2 = np.reshape(diag_prep2, [-1, 3, 3])

        rotation_1 = create_rotation_mat(list_ellipses1)
        rotation_2 = create_rotation_mat(list_ellipses2)

        cov1 = np.matmul(rotation_1, np.matmul(diag_1, np.transpose(rotation_1, (0, 2, 1))))
        cov2 = np.matmul(rotation_2, np.matmul(diag_2, np.transpose(rotation_2, (0, 2, 1))))

        h_dist_mult = np.power(np.linalg.det(cov1)[:, None, ...] * np.linalg.det(cov2), 0.25) / np.sqrt(np.linalg.det((cov1[:, None, ...] + cov2) / 2))
        h_dist_exp = -1.0 / 8 * np.matmul((mu_1[:, None, ...] - mu_2)[...,None, :], np.matmul(np.linalg.inv((cov1[:, None,...] + cov2) / 2), np.transpose((mu_1[:, None, :] - mu_2)[..., None, :], (0, 1, 3, 2))))

        return 1 - h_dist_mult * h_dist_exp


def create_rotation_mat(list_ellipse):
    '''
    Create list of rotation matrices based on list of ellipses
    :param list_ellipse:
    :return:
    '''
    theta_list = []
    phi_list = []
    for (i, ellipse) in enumerate(list_ellipse):
        theta, phi = ellipse.find_angles_fromvect()
        theta_list.append(theta)
        phi_list.append(phi)
    rotation_mat = [[[np.cos(t) * np.cos(p), -1 * np.sin(t) *
                         np.sin(p), np.sin(p)],
                        [np.sin(t)*np.cos(p), np.cos(t), np.sin(
                            t)*np.sin(p)],
                        [-np.sin(p), 0, np.cos(p)]] for (t, p) in zip(
        theta_list, phi_list)]
    return np.asarray(rotation_mat)


def create_ellipse_from_mask(seg):
    '''
    Create ellipsoid characteristics based on segmentation
    :param seg:
    :return: Ellipsoid
    '''
    #print(np.sum(seg))
    indices = np.asarray(np.where(seg == 1)).T
    indices_mean = np.mean(indices, 0)
    # demeaned = indices-indices_mean
    # cov_mat = np.cov(demeaned.T)
    if np.sum(seg) < 3:
        lambda_maj = 0
        fa = 0
        x = 0
        y = 0
        x_vec = 1
        y_vec = 1
    else:
        demeaned = indices - indices_mean
        cov_mat = np.cov(demeaned.T)
        u, s, vh = np.linalg.svd(cov_mat, full_matrices=True)
        lambda_maj = np.sqrt(s[0])
        x_vec = u[0][0]
        y_vec = u[0][1]
        mean_lambda = np.mean(np.sqrt(s))
        dist_lambda = np.sqrt(np.sum(np.square(np.sqrt(s)-mean_lambda)))
        norm_lambda = np.sqrt(np.sum(s))
        fa = np.sqrt(1.5) * dist_lambda / norm_lambda

    ell_results = Ellipsoid(x_com=indices_mean[0], y_com=indices_mean[1],
                            z_com=indices_mean[2], lambda_maj=lambda_maj,
                            fa=fa, x_vec=x_vec, y_vec=y_vec)

    # # # 2020.09.18 don't know why swopt make it better
    # ell_results = Ellipsoid(x_com=indices_mean[1], y_com=indices_mean[0],
    #                         z_com=indices_mean[2], lambda_maj=lambda_maj,
    #                         fa=fa, x_vec=x_vec, y_vec=y_vec)

    return ell_results


def nms_ellipsoid(score_map, param_map, shift=True, threshold_score=0.5,
                  threshold_overlap=0.2):
    '''
    Perform the NMS using the score map and associated params
    :param score_map:
    :param param_map: param map is in the shape [dim1, dim2, dim3, 7]
    :param shift:
    :return:
    '''


    indices_pos = np.asarray(np.where(score_map>threshold_score)).T
    scores_select = score_map[score_map>threshold_score]
    sorted_scores = -1.0 * np.sort(-1.0*scores_select)
    ind_scores = np.argsort(-1.0 * scores_select)
    param_select = []
    list_ellipses = []
    for i in ind_scores:
        index_temp = indices_pos[i,:]
        new_param = param_map[:,index_temp[0],index_temp[1],index_temp[2]]
        if shift:
            new_index = index_temp+param_map[0:3]
        else:
            new_index = index_temp
        lambda_maj = new_param[3]
        fa = new_param[4]
        x_vec = new_param[5]
        y_vec = new_param[6]
        new_ellipse = Ellipsoid(new_index[0],new_index[1],new_index[2],fa=fa,
                                lambda_maj=lambda_maj,x_vec=x_vec, y_vec=y_vec)
        list_ellipses.append(new_ellipse)
    ell = Ellipsoid()
    pw_dist = ell.hellinger_pairwise(list_ellipses)
    pw_bin = np.asarray(pw_dist<threshold_overlap)
    pw_ones = np.ones([len(list_ellipses), len(list_ellipses)])
    pw_triu = np.triu(pw_ones)
    pw_bin += pw_triu.astype(bool)
    pw_choice = np.prod(pw_bin, 1)
    indices_choice = np.where(pw_choice>0)
    ind_select = ind_scores[indices_choice]
    #fin_ellipses = list_ellipses[indices_choice]
    fin_ellipses = []
    for ii in range(np.array(indices_choice).size):
        fin_ellipses.append(list_ellipses[indices_choice[0][ii]])

    # fin_ellipses = [None]*(np.array(indices_choice).size) # Chin not sure, ask carole
    # for ii in range(np.array(indices_choice).size):
    #     fin_ellipses[ii] = list_ellipses[indices_choice[0][ii]]

    return fin_ellipses


def create_list_boxes_fromell(list_ellipses, shape):
    list_boxes = []
    list_valid = []
    for e in list_ellipses:
        #list_boxes.append(e.create_box_fromell(shape))
        sum_mask, box = e.create_box_fromell(shape)
        if sum_mask > 0:
            list_boxes.append(box)
            list_valid.append(1)
        else:
            list_valid.append(0)
    return list_boxes, list_valid



def correct_ellipses_list(list_ellipses, correction_array):
    '''
    Correct a list of ellipses based on the associated correction (N,7)
    :param list_ellipses:
    :param correction_array:
    :return:
    '''
    for (i, ell) in enumerate(list_ellipses):
        ell.correction_ellipse(correction_array[i, :])
    return list_ellipses


def create_list_ellipses(array_param_ellipses):
    '''
    Create a list of ellipses based on a array of shape N,7
    :param array_param_ellipses:
    :return:
    '''
    list_ell = []
    for i in range(array_param_ellipses.shape[0]):
        params = array_param_ellipses[i, :]
        new_ell = Ellipsoid(x_com=params[0], y_com=params[1], z_com=params[2], lambda_maj=params[3], fa=params[4], x_vec=params[5], y_vec=params[6])
        list_ell.append(new_ell)
    return list_ell