from skimage.measure import regionprops
from skimage import draw
from skimage import graph, data, io, segmentation, color, filters
import skimage.future.graph as graph

import copy
import numpy as np

class RAG ():
    def __init__ (self, args, image, label, n_segments=1024, compactness=0.1, split_r=2):
        self.image = image
        self.label = label # Indexing from 1, background is 1
        self.n_segments = n_segments
        self.compactness = compactness
        self.split_r = split_r
        self.init_graph ()
        self.extend_graph ()

    def init_graph (self):
         # Over segmentations
        self.segments = segmentation.slic(self.image, compactness=0.1, n_segments=self.n_segments, start_label=1)
        # Edge map from sobel filter
        self.edge_map = filters.sobel(color.rgb2gray(self.image))
        # Region Adjacency Graph by boundary 
        self.rag = graph.rag_boundary(self.segments, self.edge_map)
        # Get each node with properties
        self.nodes = self.rag.nodes
        self.regions = regionprops(label_image=self.segments, intensity_image=self.label)
        self.n_segments = len (self.nodes)

        # Construct binary oversegment map [0, 255] (HxWx1)
        bin_map = np.ones (self.edge_map.shape, dtype=np.uint8) * 255 
        self.boundary_map = segmentation.mark_boundaries(bin_map, self.segments, 0) [:,:,:1]

        # Build adjacency edge lists
        self.adj_edges = [[] for i in range (self.n_segments + 1)]
        self.edges = []
 
        for edge in self.rag.edges:
            u, v = edge
            self.adj_edges [u].append (v)
            self.adj_edges [v].append (u)
            self.edges.append ((u, v))

    def extend_graph (self):
        edges = list (self.rag.edges)
        
        ''' Note:
        # region ['intensity_image'] : label map, background = 1, objects > 1
        # region ['image'] : binary segment
        '''

        # Initial mapping for each node
        self.node_colors = [0] * (np.max (self.segments) + 1)

        lbl_ids, ins_area = np.unique (self.label, return_counts=True)
        self.N_Instances = len (lbl_ids)

        # Initialize body segments list of each ground truth instance
        self.instances = [[] for i in range (self.N_Instances + 1)]
        # Initiallze body segments' affinities
        self.ins_affinities = [[] for i in range (self.N_Instances + 1)]
        # Total area of instance
        self.ins_area = [0 for i in range (self.N_Instances + 1)]
        # Initialize neighbor segments list of each ground truth instance
        self.neighbors = [[] for i in range (self.N_Instances + 1)]
        # Initialize neighbor segments' affinies
        self.nei_affinities = [[] for i in range (self.N_Instances + 1)]
        # Total area of neighbor segments
        self.nei_area = [0 for i in range (self.N_Instances + 1)]

        nodes = self.rag.nodes

        for lbl_id, area in zip (lbl_ids, ins_area):
            self.ins_area [lbl_id] = area

        # Get properties from regionprop
        for region in self.regions:
            idx = region ['label']
            nodes[idx]['bbox'] = region ['bbox']
            nodes[idx]['foreground_area'] = np.count_nonzero (region ['image'] * (region ['intensity_image'] != 1))
            nodes[idx]['background_area'] = np.count_nonzero (region ['image'] * (region ['intensity_image'] == 1))
            nodes[idx]['label_image'] = region ['intensity_image']
            nodes[idx]['centroid'] = region ['centroid']

            # Update foreground instances of each segment (values, counts of values)
            nodes[idx]['instances'], nodes[idx]['affinity'] = \
                        np.unique (region ['intensity_image'] * region ['image'] * (region ['intensity_image'] != 1), return_counts=True)

            # Remove non-foreground values of each segment (the step above generates 0 labels (where label == 1))
            nodes [idx]['affinity'] = np.delete (nodes [idx]['affinity'], np.where (nodes[idx]['instances']==0))
            nodes [idx]['instances'] = np.delete (nodes [idx]['instances'], np.where (nodes[idx]['instances']==0))

            # Normalize to 0~1 affinity
            nodes[idx]['affinity'] = nodes[idx]['affinity'] / nodes[idx]['foreground_area']

            # Update background affinity of each segment
            nodes[idx]['bg_affinity'] = nodes[idx]['background_area'] / region ['area']

        # Update list of segments for each instance (Used for reward computation of each instance)
        for region in self.regions:
            idx = region ['label']
            for i, lbl_id in enumerate (nodes [idx]['instances']):
                self.instances [lbl_id].append (idx)
                self.ins_affinities [lbl_id].append (nodes[idx]['affinity'][i])

        # For each instances, store the list of body segments and affinities as 1D numpy vectors
        for lbl_id in range (self.N_Instances + 1):
            self.instances [lbl_id] = np.array (self.instances [lbl_id])
            self.ins_affinities [lbl_id] = np.array (self.ins_affinities [lbl_id])

        # Update list of neighbors for each instance
        for region in self.regions:
            # Each segment
            idx = region ['label']
            node_instances = nodes [idx]['instances']
            node_affinity = nodes [idx]['affinity']
            if len (node_instances) > 1:
                # Borderline segment, adjacent to multiple objects
                for aff, u in zip (node_affinity, node_instances):
                    self.neighbors [u].append (idx)
                    self.nei_affinities [u].append (aff)
                    self.nei_area [u] += aff * nodes [idx]['foreground_area']

        self.new_edges = [[] for i in range (self.N_Instances + 1)]

        # For each instance, BFS to find the neighboring segments
        for ins_id, segments in enumerate (self.instances):
            # Should have no segments in ins_id = 0 (instance label >= 1)
            if (ins_id == 0):
                continue

            new_neighbors = []
            new_edges = []
            # Distance from the segment
            distance = [0] * len (segments)
            # Copy the segments list and cast to list 
            segments = segments.tolist ()


            for d, u in zip (distance, segments):
                if d + 1 > self.split_r:
                    continue
                for v in self.adj_edges[u]:
                    # For each v incident from u
                    if (v not in self.neighbors[ins_id]) \
                            and (v not in segments):
                        # If not already in the label list and not in the body segment
                        # and not in newly discovered neighbors
                        new_neighbors.append (v)
                        segments.append (v)
                        distance.append (d + 1)
                        new_edges.append ((u,v))
                        self.nei_area [ins_id] += nodes [v]['foreground_area']

            # Store neighbors list and affinities in 1D numpy arrays
            self.neighbors [ins_id].extend (new_neighbors)
            self.neighbors [ins_id] = np.array (self.neighbors [ins_id])
            self.nei_affinities [ins_id].extend ([1.0] * len(new_neighbors))
            self.nei_affinities [ins_id] = np.array (self.nei_affinities [ins_id])
            # Store edges that connect instance to neighboring segments
            self.new_edges [ins_id].extend (new_edges)

        self.nei_area = np.array (self.nei_area)
        self.ins_area = np.array (self.nei_area)


    def vec2img (self, values):
        # Transform 1D vector -> 2D image
        # values must be 1D numpy array of length of self.n_segments + 1 (segment index start from 1)
        assert (len (values) == self.n_segments + 1)
        return values [self.segments]


                        



