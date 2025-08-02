#ifndef EDGE_CLUSTERER_HPP
#define EDGE_CLUSTERER_HPP

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <memory>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "definitions.h"
#include "toed/cpu_toed.hpp"
#include "utility.h"

//> Custom hash function for std::pair<int, int>
struct PairHash {
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& p) const {
        size_t h1 = std::hash<T1>{}(p.first);
        size_t h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

class EdgeClusterer {
public:

    //> Constructor
    EdgeClusterer( std::vector<Edge> );

    //> main code for edge clustering
    void performClustering( );

    // Eigen::MatrixXd Epip_Correct_Edges;
    std::vector<Edge> Epip_Correct_Edges;
    std::vector<int> cluster_labels;
    std::vector<std::vector<int> > clusters;
    std::unordered_map<int, double> cluster_avg_orientations;
    unsigned Num_Of_Clusters;
    
    //> For each edge index, store all other edge indices in the same cluster
    // std::unordered_map<std::pair<int, int>, std::vector<int>, PairHash> H2_Clusters; //<H1 edge index, H2 edge index>, cluster of H2 edges

private:
    int getClusterSize(int label);
    double normalizeOrientation(double orientation);
    std::tuple<double, double, double> computeGaussianAverage( int label1, int label2 = -1 );

    int H1_edge_idx;
    int Num_Of_Epipolar_Corrected_H2_Edges;
};

#endif // EDGE_CLUSTERER_HPP
