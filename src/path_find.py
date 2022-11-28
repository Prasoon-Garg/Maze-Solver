import numpy as np
from matplotlib import pyplot as plt


def create_graph(image):
    h = image.shape[0]
    w = image.shape[1]
    adj_list = [[] for i in range(h*w)]
    for i in range(h):
        for j in range(w):
            if image[i][j] == 255:
                if i>0 and image[i-1][j] == 255:
                    adj_list[(i*w)+j].append((i-1)*w + j)
                if j>0 and image[i][j-1] == 255:
                    adj_list[(i*w)+j].append(i*w + j - 1)
                if i<h-1 and image[i+1][j] == 255:
                    adj_list[(i*w)+j].append((i+1)*w + j)
                if j<w-1 and image[i][j+1] == 255:
                    adj_list[(i*w)+j].append(i*w + j + 1)
    return adj_list


def BFS(adj, src, dest, v, pred, dist):
 
    queue = []
  
    visited = [False for i in range(v)]
  
    for i in range(v):
 
        dist[i] = 1000000
        pred[i] = -1
     
    visited[src] = True
    dist[src] = 0
    queue.append(src)
  
    while (len(queue) != 0):
        u = queue[0]
        queue.pop(0)
        for i in range(len(adj[u])):
         
            if (visited[adj[u][i]] == False):
                visited[adj[u][i]] = True
                dist[adj[u][i]] = dist[u] + 1
                pred[adj[u][i]] = u
                queue.append(adj[u][i])
  
                if (adj[u][i] == dest):
                    return True
  
    return False


def ShortestDistance(adj, s, dest):
    
    v = len(adj)
    
    pred=[0 for i in range(v)]
    dist=[0 for i in range(v)]
  
    if (BFS(adj, s, dest, v, pred, dist) == False):
        print("Given source and destination are not connected")
        
    else:
        path = []
        crawl = dest
        path.append(crawl)

        while (pred[crawl] != -1):
            path.append(pred[crawl])
            crawl = pred[crawl]

        return path

def final_path(image,src,destin):

    source = (src[0] * image.shape[1]) + src[1]
    dest = (destin[0] * image.shape[1]) + destin[1]

    adjacency_list = create_graph(image)
    return ShortestDistance(adjacency_list,source,dest)