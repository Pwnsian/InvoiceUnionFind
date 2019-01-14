#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iterator>
#include <unordered_map>

#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"
#include  "opencv2/features2d.hpp"

#include "dset.h"

typedef std::vector<cv::Point> MSERResult;

float omega = 1.4;
float alpha = 1.2;
float phi = 0.1;
//float gamma = 6;

class Component
{
public:
    Component()
    : width(0)
    , height(0)
    , averageGrayLevel(0)
    {

    }

    Component(const MSERResult& mser)
    {
        boundingRect = cv::boundingRect(mser);
        width = boundingRect.width;
        height = boundingRect.height;
        pixelList = std::move(mser);
        averageGrayLevel = mser.size() / boundingRect.area();
        centroid = computeCentroid();
    }

    bool shouldCombine(const Component& other) const
    {
        //auto cent1 = cv::Rect(boundingRect.x, boundingRect.y, boundingRect.width / 2, boundingRect.height / 2);
        //auto cent2 = cv::Rect(other.boundingRect.x, other.boundingRect.y, other.boundingRect.width / 2, other.boundingRect.height / 2);

        auto cent1 = cv::Point(boundingRect.x + boundingRect.width / 2, boundingRect.y + boundingRect.height / 2);
        auto cent2 = cv::Point(other.boundingRect.x + other.boundingRect.width / 2, other.boundingRect.y + other.boundingRect.height / 2);

        //auto cent1 = centroid;
        //auto cent2 = other.centroid;

        bool proximityXOk = std::abs(cent1.x - cent2.x) < ((width * omega) + (other.width * omega));
        bool proximityYOk = std::abs(cent1.y - cent2.y) < ((height * omega * 0.75) + (other.height * omega * 0.75));
        if(!proximityXOk || !proximityYOk)
            return false;

        bool xRatioOk = std::min(width,other.width) / std::max(width, other.width) <= alpha;
        bool yRatioOk = std::min(height,other.height) / std::max(height, other.height) <= alpha;
        if(!xRatioOk || !yRatioOk)
            return false;

        bool averageGrayOk = std::abs(averageGrayLevel - other.averageGrayLevel) <= phi;
        if(!averageGrayOk)
            return false;

        return true;
    }

    cv::Rect getBoundingRect() const
    {
        return boundingRect;
    }

    const MSERResult& getContour()
    {
        return pixelList;
    }

private:
    cv::Point computeCentroid() const
    {
        cv::Moments m = moments(pixelList, true);
        cv::Point center(m.m10/m.m00, m.m01/m.m00);
        return center;
    }

private:
    // Paper-specified features
    int width;
    int height;
    int averageGrayLevel;
    MSERResult pixelList;

    // Helper items
    cv::Point centroid;
    cv::Rect boundingRect;
};

cv::Rect Union(const cv::Rect& a, const cv::Rect& b) { 
    int x1 = std::min(a.x, b.x);
    int x2 = std::max(a.x + a.width, b.x + b.width); 
    int y1 = std::min(a.y, b.y);
    int y2 = std::max(a.y + a.height, b.y + b.height);
    return cv::Rect(x1, y1, x2 - x1, y2 - y1); 
}

int main( int argc, char** argv )
{
    cv::Mat img, gray;
    img = cv::imread(argv[1]);
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    int delta = 13;
    int img_area = gray.cols * gray.rows;
    cv::Ptr<cv::MSER> cv_mser = cv::MSER::create();
    //cv_mser->setMinArea(50);
    //cv_mser->setMaxArea(1000);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Rect>  mser_bboxes;
    cv_mser->detectRegions(gray, contours, mser_bboxes);



    std::vector<Component> components;
    std::transform(contours.begin(), contours.end(), std::back_inserter(components), [](MSERResult& mser) {
        return Component(mser);
    });

    std::sort(components.begin(), components.end(), [](const Component& a, const Component& b) {
        return a.getBoundingRect().x < b.getBoundingRect().x;
    });

    DisjointSets paragraphs(components.size());

    cv::Mat debug = img.clone();
    for(const auto pair : mser_bboxes)
    {
        cv::Scalar color(255, 255, 0);
        cv::rectangle(debug, pair, color, 5);
    }
    cv::imwrite("output.png", debug);


    for(int i = 0; i < components.size(); ++i)
    for(int j = 0; j < components.size(); ++j)
    {
        if(i == j)
            continue;
        const Component& a = components[i];
        const Component& b = components[j];
        if(a.shouldCombine(b))
        {
            int parentId = paragraphs.unite(i,j);
        }
    }

    std::cout << paragraphs << std::endl;

    std::unordered_map<int, cv::Rect> parentedRects;
    for (size_t i=0; i < paragraphs.mData.size(); ++i)
    {
        int parentId = paragraphs.parent(i);
        if(parentedRects.count(parentId))
        {
            parentedRects[parentId] = Union(parentedRects[parentId], components[i].getBoundingRect());
        }
        else
        {
            parentedRects[parentId] = components[parentId].getBoundingRect();
        }
    }

    cv::Mat finalImg = img.clone();
    for(const auto pair : parentedRects)
    {
        cv::Scalar color(255, 0, 0);
        cv::rectangle(finalImg, pair.second, color, 5);
    }

    std::vector<cv::Scalar> a = {{43, 43, 200}, {43, 75, 200}, {43, 106, 200}, {43, 137, 200}, {43, 169, 200}, {43, 200, 195}, {43, 200, 163}, {43, 200, 132}, {43, 200, 101}, {43, 200, 69}, {54, 200, 43}, {85, 200, 43}, {116, 200, 43}, {148, 200, 43}, {179, 200, 43}, {200, 184, 43}, {200, 153, 43}, {200, 122, 43}, {200, 90, 43}, {200, 59, 43}, {200, 43, 64}, {200, 43, 95}, {200, 43, 127}};    

    for(int i = 0; i < components.size(); ++i)
    {
        int parentId = paragraphs.parent(i);
        cv::Scalar color = a[parentId % a.size()];
        std::vector<MSERResult> contoursArray = { components[i].getContour() };
        cv::drawContours(finalImg, contoursArray, -1, color);
        //cv::imwrite("final" + std::to_string(i) + ".png", finalImg);
    }
    cv::imwrite("final.png", finalImg);
}