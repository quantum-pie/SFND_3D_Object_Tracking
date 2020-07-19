
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, topviewImg);
    cv::resizeWindow(windowName, 800, 600);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBoxPrev, BoundingBox &boundingBoxCurr, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    double mean = 0;
    double stdev = 0;
    std::vector<std::pair<double, cv::DMatch>> dists;
    for (const auto &match : kptMatches) {
        const auto &prev_pt = kptsPrev[match.queryIdx].pt;
        const auto &curr_pt = kptsCurr[match.trainIdx].pt;
        if (boundingBoxPrev.roi.contains(prev_pt) and boundingBoxCurr.roi.contains(curr_pt)) {
            // If we assume that keypoints distances normally distributed, then
            // distance between points has Rayleigh distribution. This distribution is
            // assymmetric and we need to normalize it to reliably filter outliers.
            // We can use Wilson-Hilferty transformation for this purpose: Y = X^(2/3)
            auto dist = std::pow(cv::norm(curr_pt - prev_pt), 2.0 / 3); 
            mean += dist;
            dists.emplace_back(dist, match);
        }
    }
    
    if (dists.empty()) {
        return;
    }
    
    mean /= dists.size();
    // calculate standard deviation
    for (const auto &pt : dists) {
        stdev += std::pow((pt.first - mean), 2.0);
    }
    stdev = std::sqrt(stdev / dists.size());
    
    // sort to find robust mean - median
    std::sort(dists.begin(), dists.end(), [](auto &&lhs, auto &&rhs){
        return lhs.first < rhs.first;
    });
    
    // filter outliers
    auto median = dists[dists.size() / 2].first;
    for (const auto &pt : dists) {
        if (std::fabs(median - pt.first) < 1.5 * stdev) {
            boundingBoxCurr.kptMatches.push_back(pt.second);
        }
    }
    
    //std::cout << "Unfiltered matches: " << dists.size() << '\n';
    //std::cout << "Filtered matches: " << boundingBoxCurr.kptMatches.size() << '\n';
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            //&& std::fabs(distPrev - distCurr) > std::numeric_limits<double>::epsilon() )
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence
    //std::cout << "Number of ratios: " << distRatios.size() << '\n';
    //std::cout << "Median: " << medDistRatio << '\n';
    
    TTC = 1.0 / (frameRate * (medDistRatio - 1.0));
}

static double closestClusterDist(std::vector<LidarPoint> &cloud) {
    auto ComparatorX = [](auto &&lhs, auto &&rhs){
       return lhs.x < rhs.x; 
    };
    
    // Sort lidar points by x-coordinate = distance from the ego car
    std::sort(cloud.begin(), cloud.end(), ComparatorX);
    static constexpr size_t min_cluster = 20;
    static constexpr double dist_tol = 0.001;
    size_t cluster_size = 1;
    
    // Start from the closest point to car and try to build 
    // cluster around it based on the x-coordinate neighborhood.
    // Distance to the pointcloud is a distance to the closest point of
    // the first found cluster.
    // This algorithm is much faster than generic NN-clustering in 3D space,
    // beacause it builds only one cluster and utilizes importance of only x-coordinate.
    for (auto it = std::next(cloud.begin()); it != cloud.end(); ++it) {
        auto prev_it = std::prev(it);
        if (it->x - prev_it->x < dist_tol) {
            if (++cluster_size == min_cluster) {
                return std::prev(it, min_cluster - 1)->x;
            }
        } else {
            cluster_size = 1;
        }
    }
    // Return closest point if failed to build cluster
    return cloud.front().x;
}

static double centroid(std::vector<LidarPoint> &cloud) {
    auto ComparatorX = [](auto &&lhs, auto &&rhs){
       return lhs.x < rhs.x; 
    };
    
    // find median of pointcloud along x axis (median distance to the ego car)
    std::sort(cloud.begin(), cloud.end(), ComparatorX);
    auto idx = cloud.size() / 2;
    return cloud.size() % 2 == 0 ? (cloud[idx - 1].x + cloud[idx].x) / 2.0 : cloud[idx].x; 
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // Try to make accurate estimate of current distance using clustering 
    auto curr_dist = closestClusterDist(lidarPointsCurr);
    
    // Calculate centroids of current and previous pointclouds along x axis.
    // This operation allows to make the most accurate estimate of the translation
    // between previous and current positions of a car. 
    // We use median instead of closest cluster distance, because for calculation
    // of delta distance (previous - current) we can choose any anchor point 
    // in the pointcloud we want. We assume that median is the most stable and robust anchor
    // to use in this particular case. It is also can be demonstatred in error analysis -
    // presented approach is more accurate and stable compared to approach using closest cluster 
    // distance to calculate delta
    auto prev_centroid = centroid(lidarPointsPrev);
    auto curr_centroid = centroid(lidarPointsCurr);
    
    if (std::fabs(prev_centroid - curr_centroid) > std::numeric_limits<double>::epsilon()) {
        TTC = curr_dist / (frameRate * (prev_centroid - curr_centroid));
    } else {
        TTC = NAN;
    }
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    struct Count {
        int n = 0;
    };
    
    // This double map hold correspondence prevBox -> currentBox -> number of voted points
    std::map<int, std::map<int, Count>> matching_candidates;
    
    // This map holds correspondence prevBox -> current maximal number of voted points
    std::map<int, Count> best_counts;
    
    // Proposed algorithm uses only one pass through matches vector and
    // do not use cross iteration over previous and current bounding boxes
    for (const auto & match : matches) {
        // find all enclosing boxes in current frame
        std::vector<int> curr_frame_enclosing_boxes;
        for (const auto& box : currFrame.boundingBoxes) {
            if (box.lidarPoints.empty()) {
                continue;
            }
            if (box.roi.contains(currFrame.keypoints[match.trainIdx].pt)) {
                curr_frame_enclosing_boxes.push_back(box.boxID);
            }
        }
        
        // iterate through previous frame boxes
        for (const auto& box : prevFrame.boundingBoxes) {
            if (box.lidarPoints.empty()) {
                continue;
            }
            auto prev_box_id = box.boxID;
            auto &best_count = best_counts[prev_box_id].n;
            if (box.roi.contains(prevFrame.keypoints[match.queryIdx].pt)) {
                // this point votes for a match
                auto &prev_box_map = matching_candidates[prev_box_id];
                for (auto curr_box_id : curr_frame_enclosing_boxes) {
                    auto& count = prev_box_map[curr_box_id].n;
                    ++count;
                    // refresh best match if number of votes is more than the current maximum
                    if (count > best_count) {
                        best_count = count;
                        bbBestMatches[prev_box_id] = curr_box_id;
                    }
                }
            }
        }
    }
}
