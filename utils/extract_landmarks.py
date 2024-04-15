import cv2
from face_alignment import FaceAlignment, LandmarksType
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from scipy.spatial import Delaunay
import torch
import math

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def detect_landmark(image_path, classe):
    # Load the image
    image = cv2.imread(image_path)

    # Initialize FaceAlignment with FAN model
    fa = FaceAlignment(LandmarksType.TWO_D, device='cuda')

    # Detect landmarks
    landmarks = fa.get_landmarks(image)
    landmarks_ = landmarks[0]
    used_points = [17,18,19,20,21,
                   22,23,24,25,26,
                   36,37,38,39,40,
                   41,42,43,44,45,
                   46,47,30,48,49,
                   50,51,52,53,54,
                   55,56,57,58,59,
                   60,61,62,63,64,
                   65,66,67]
    landmarks_ = [landmarks_[i] for i in used_points]
    triangulation = Delaunay(landmarks_)

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cont = 0
    color = 'white'
    line_width = 0.5
    edges = list(); dist = list()
    for simplex in triangulation.simplices:
        a_, b_, c_ = simplex
        
        dist_a_b = calculate_distance(landmarks_[a_][0], landmarks_[a_][1], landmarks_[b_][0], landmarks_[b_][1])
        dist_a_c = calculate_distance(landmarks_[a_][0], landmarks_[a_][1], landmarks_[c_][0], landmarks_[c_][1])
        dist_b_c = calculate_distance(landmarks_[b_][0], landmarks_[b_][1], landmarks_[c_][0], landmarks_[c_][1])
        
        edges.append(torch.LongTensor([a_, b_]))
        dist.append(dist_a_b)
        edges.append(torch.LongTensor([b_, a_]))
        dist.append(dist_a_b)
        edges.append(torch.LongTensor([a_, c_]))
        dist.append(dist_a_c)
        edges.append(torch.LongTensor([c_, a_]))
        dist.append(dist_a_c)
        edges.append(torch.LongTensor([b_, c_]))
        dist.append(dist_b_c)
        edges.append(torch.LongTensor([c_, b_]))
        dist.append(dist_b_c)
        
        # plt.plot((landmarks_[a_][0], landmarks_[b_][0]),(landmarks_[a_][1], landmarks_[b_][1]), 'k-', color=color, linewidth=line_width)
        # plt.plot((landmarks_[a_][0], landmarks_[c_][0]),(landmarks_[a_][1], landmarks_[c_][1]), 'k-', color=color, linewidth=line_width)
        # plt.plot((landmarks_[b_][0], landmarks_[c_][0]),(landmarks_[b_][1], landmarks_[c_][1]), 'k-', color=color, linewidth=line_width)

    color2 = 'dodgerblue'
    nodes = list()
    
    for landmark in landmarks_:
        nodes.append(torch.FloatTensor([landmark[0], landmark[1]]))
        # plt.scatter(landmark[0], landmark[1], s=25, c=color2, marker='o', zorder=10)
        cont += 1
        # plt.axis('off')
        
    nodes = torch.stack(nodes)
    edges = torch.stack(edges)
    edges = edges.t()
    dist = torch.FloatTensor(dist)
    dado = Data(x=nodes, edge_index=edges, edge_attr=dist, y=classe)
    return dado